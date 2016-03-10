#!/usr/bin/env python

# Designed for Python 2 (will NOT run on Python 3)

import argparse
import pycuda.driver as cuda
import numpy
import matplotlib.pyplot as plt
import ishne
import sys
import timer
import custom_functions
import multiprocessing
import os

def load_CUDA():
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    with open("plotecg.cu") as wavelet_file:
        mod = SourceModule(wavelet_file.read())

    global mexican_hat
    global cross_correlate_with_wavelet
    global threshold
    global edge_detect
    global filter
    global median
    global get_rr
    global index_of_peak
    global merge_leads
    global nonzero
    global scatter
    global to_float
    global get_compact_rr
    global moving_average
    global clean_result

    mexican_hat = mod.get_function("mexican_hat")
    cross_correlate_with_wavelet = mod.get_function("cross_correlate_with_wavelet")
    threshold = mod.get_function("threshold")
    edge_detect = mod.get_function("edge_detect")
    filter = mod.get_function("int_3_median_filter")
    median = mod.get_function("int_3_median_reduction")
    get_rr = mod.get_function("get_rr")
    index_of_peak = mod.get_function("index_of_peak")
    merge_leads = mod.get_function("merge_leads")
    nonzero = mod.get_function("nonzero")
    scatter = mod.get_function("scatter")
    to_float = mod.get_function("to_float")
    get_compact_rr = mod.get_function("get_compact_rr")
    moving_average = mod.get_function("moving_average")
    clean_result = mod.get_function("clean_result")

def moving_average_filter(dev_array, length, window):
    scan_result = cuda.mem_alloc(length * 4)
    custom_functions.exclusive_scan(scan_result, dev_array, length)
    grid = ((length / 1024) + 1, 1)
    block = (1024, 1, 1)
    moving_average(dev_array, scan_result,
                   numpy.int32(window), numpy.int32(length),
                   grid=grid, block=block)

def compress_leads(*leads):
    return tuple(custom_functions.turning_point_compression(lead, times=2).astype(numpy.float16)
                 for lead in leads)

def transfer_leads(*h_leads):
    length = len(h_leads[0])
    result = []
    grid = ((length / 1024)+1, 1)
    block = (1024, 1, 1)
    for h_lead in h_leads:
        d_lead16 = cuda.to_device(h_lead)
        d_lead32 = cuda.mem_alloc(h_lead.nbytes * 2)
        to_float(d_lead32, d_lead16, numpy.int32(length),
                 grid=grid, block=block)
        result.append(d_lead32)
    return tuple(result) + (length,)

def generate_hat(num_samples):
    # The math suggests 16 samples is the width of the QRS complex
    # Measuring the QRS complex for 9004 gives 16 samples
    # Measured correlated peak 7 samples after start of QRS
    # Mexican hats seem to hold a nonzero value between -4 and 4 w/ sigma=1
    sigma = 1.0
    maxval = 4 * sigma
    minval = -maxval

    hat = numpy.zeros(num_samples).astype(numpy.float32)
    mexican_hat(cuda.Out(hat),
                numpy.float32(sigma),
                numpy.float32(minval),
                numpy.float32((maxval - minval)/num_samples),
                grid=(1, 1), block=(num_samples, 1, 1))
    return hat

def median_filter(out_array, in_ary, grid, block):
    padded = numpy.pad(in_ary, (1, 1), mode="edge")
    filter(cuda.Out(out_array), cuda.In(padded), grid=grid, block=block)
    return out_array

# Note: Inlining this saves 50ms per invocation
def preprocess_lead(d_lead, lead_size, d_wavelet,
                    wavelet_len, threshold_value):
    # Kernel Parameters
    threads_per_block = 200
    num_blocks = lead_size / threads_per_block

    # correlate lead with wavelet
    correlated = cuda.mem_alloc(lead_size * 4)
    cross_correlate_with_wavelet(correlated, d_lead, d_wavelet,
                                 numpy.int32(lead_size),
                                 numpy.int32(wavelet_len),
                                 grid=(num_blocks, 1),
                                 block=(threads_per_block, 1, 1))

    # threshold correlated lead
    thresholded_signal = cuda.mem_alloc(lead_size * 4)
    threshold(thresholded_signal, correlated,
              numpy.float32(threshold_value),
              grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    return thresholded_signal

def preprocess(d_lead1, d_lead2, d_lead3, lead_size,
               d_wavelet, wavelet_len, threshold_value, sampling_rate):
    d_tlead1 = preprocess_lead(d_lead1,
                               lead_size,
                               d_wavelet,
                               wavelet_len,
                               threshold_value)
    d_tlead2 = preprocess_lead(d_lead2,
                               lead_size,
                               d_wavelet,
                               wavelet_len,
                               threshold_value)
    d_tlead3 = preprocess_lead(d_lead3,
                               lead_size,
                               d_wavelet,
                               wavelet_len,
                               threshold_value)

    # synchronize & merge
    d_merged_lead, lead_len = synchronize_and_merge(d_tlead1,
                                                    d_tlead2,
                                                    d_tlead3,
                                                    lead_size,
                                                    sampling_rate)
    return (d_merged_lead, lead_len)

def synchronize_and_merge(d_tlead1, d_tlead2, d_tlead3, length, sampling_rate):
    (offset1, offset2, offset3, lead_len) = synchronize(d_tlead1,
                                                        d_tlead2,
                                                        d_tlead3,
                                                        length,
                                                        sampling_rate)
    # merge
    d_merged_lead, lead_len = merge(d_tlead1, offset1, d_tlead2, offset2,
                                    d_tlead3, offset3, lead_len)
    return (d_merged_lead, lead_len)    

def cpu_synchronize(lead1, lead2, lead3, length):
    start1 = numpy.argmax(lead1)
    start2 = numpy.argmax(lead2)
    start3 = numpy.argmax(lead3)
    minstart = min(start1, start2, start3)
    maxstart = max(start1, start2, start3)
    offset1 = start1 - minstart
    offset2 = start2 - minstart
    offset3 = start3 - minstart
    new_length = length - (maxstart - minstart)
    return (offset1, offset2, offset3, new_length)

def synchronize(d_tlead1, d_tlead2, d_tlead3, length, sampling_rate):
    # Number of points to use to synchronize
    chunk = sampling_rate * 2
    template = numpy.zeros(chunk).astype(numpy.int32)
    tlead1 = cuda.from_device_like(d_tlead1, template)
    tlead2 = cuda.from_device_like(d_tlead2, template)
    tlead3 = cuda.from_device_like(d_tlead3, template)
    start1 = numpy.argmax(tlead1)
    start2 = numpy.argmax(tlead2)
    start3 = numpy.argmax(tlead3)
    minstart = min(start1, start2, start3)
    maxstart = max(start1, start2, start3)
    offset1 = start1 - minstart
    offset2 = start2 - minstart
    offset3 = start3 - minstart
    new_length = length - (maxstart - minstart)
    return (offset1, offset2, offset3, new_length)

def merge(d_slead1, offset1, d_slead2, offset2, d_slead3, offset3, length):
    # Kernel Parameters
    threads_per_block = 200
    num_blocks = length / threads_per_block

    d_merged_lead = cuda.mem_alloc(4 * num_blocks * threads_per_block)
    merge_leads(d_merged_lead,
                d_slead1, numpy.int32(offset1),
                d_slead2, numpy.int32(offset2),
                d_slead3, numpy.int32(offset3),
                grid=(num_blocks, 1), block=(threads_per_block, 1, 1))
    return d_merged_lead, num_blocks * threads_per_block

def get_heartbeat(d_lead, length, sampling_rate):
    # Kernel Parameters
    threads_per_block = 200
    num_blocks = length / threads_per_block


    # Get RR
    reduce_by = 32
    edge_signal = cuda.mem_alloc(4 * length)
    
    edge_detect(edge_signal, d_lead,
                grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    indecies = numpy.zeros(length / reduce_by).astype(numpy.int32)
    masks = cuda.to_device(numpy.zeros(length / reduce_by).astype(numpy.int32))
    d_index = cuda.to_device(indecies)
    index_of_peak(d_index, masks, edge_signal,
                  grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    cd_index, c_length = compact_sparse_with_mask(d_index, masks, length / reduce_by)

    # Allocate output
    # full_rr_signal = numpy.zeros(c_length).astype(numpy.int32)
    dev_rr = cuda.mem_alloc(c_length * 4)

    num_blocks = (c_length / threads_per_block) + 1
    get_compact_rr(dev_rr,
                   cd_index,
                   numpy.int32(sampling_rate),
                   numpy.int32(c_length),
                   grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    clean_result(dev_rr, numpy.int32(120), numpy.int32(40),
                 numpy.int32(1), numpy.int32(c_length),
                 grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    moving_average_filter(dev_rr, c_length, 250)

    index = cuda.from_device(cd_index, (c_length,), numpy.int32)
    rr = cuda.from_device(dev_rr, (c_length,), numpy.int32)
    index[0] = index[1]

    return rr, index / float(sampling_rate * 3600)

def compact_sparse(dev_array, length):
    contains_result = cuda.mem_alloc(length * 4)
    block_size = 64
    if length % block_size:
        grid_size = (length / block_size) + 1
    else:
        grid_size = (length / block_size)
    grid = (grid_size, 1)
    block = (block_size, 1, 1)
    nonzero(contains_result, dev_array, numpy.int32(length), grid=grid, block=block)
    return compact_sparse_with_mask(dev_array, contains_result, length)

def compact_sparse_with_mask(dev_array, dev_mask, length):
    block_size = 64
    if length % block_size:
        grid_size = (length / block_size) + 1
    else:
        grid_size = (length / block_size)
    grid = (grid_size, 1)
    block = (block_size, 1, 1)
    scan_result = cuda.mem_alloc(length * 4)
    custom_functions.exclusive_scan(scan_result, dev_mask, length)
    new_length = custom_functions.index(scan_result, length-1)
    result = cuda.mem_alloc(new_length * 4)
    scatter(result, dev_array, scan_result, dev_mask, numpy.int32(length), grid=grid, block=block)
    scan_result.free()
    dev_mask.free()
    return result, new_length

def read_ISHNE(ecg_filename):
    # Read the ISHNE file
    ecg = ishne.ISHNE(ecg_filename)
    ecg.read()
    return ecg

def plot_leads(ecg_filename, lead_numbers):

    ecg = read_ISHNE(ecg_filename)
    num_seconds = 5
    num_points = ecg.sampling_rate * num_seconds
    plt.figure(1)
    for lead_number in lead_numbers:
        if lead_number > len(ecg.leads):
            print "Error: ECG does not have a lead", lead_number
            return
        x = numpy.linspace(0, num_seconds, num=num_points)
        y = ecg.leads[lead_number - 1][:num_points]
        plt.plot(x, y)
    plt.title("ECG")
    plt.xlabel("Seconds")
    plt.ylabel("mV")
    plt.show()

def get_hr(compressed_leads, sampling_rate):
    # number of samples: 0.06 - 0.1 * SAMPLING_RATE (QRS Time: 60-100ms)
    num_samples = int(0.08 * sampling_rate) + 2
    with timer.GPUTimer(cuda) as hatgen:
        wavelet = generate_hat(num_samples)
        d_wavelet = cuda.to_device(wavelet)
    wavelet_len = len(wavelet)

    with timer.GPUTimer(cuda) as transfer:
        d_lead1, d_lead2, d_lead3, length = transfer_leads(*compressed_leads)
    with timer.GPUTimer(cuda) as pre:
        d_mlead, length_mlead = preprocess(d_lead1, d_lead2, d_lead3,
                                           length, d_wavelet, wavelet_len,
                                           0.5, sampling_rate)
    with timer.GPUTimer(cuda) as rr:
        heartrate = get_heartbeat(d_mlead, length_mlead, sampling_rate)
    print "GPU Compute:", transfer, "(transfer)", pre, "(preprocess)", rr, "(process)"
    return heartrate

def plot_hr(ecg_filename):

    load_CUDA()

    ecg = read_ISHNE(ecg_filename)

    with timer.GPUTimer(cuda) as compression:
        compressed_leads = compress_leads(*ecg.leads)

    print "Compression:", compression

    with timer.GPUTimer(cuda) as compute:
        y, x = get_hr(compressed_leads, ecg.sampling_rate / 4)

    print "HR processed in", compute.interval + compression.interval, "ms"

    cuda.Context.synchronize()
    plt.figure(1)
    plt.plot(x, y)
    plt.title("ECG - RR")
    plt.xlabel("Hours")
    plt.ylabel("Heartrate (BPM)")
    plt.show()

def plot_hr_many(filenames):
    load_CUDA()
    ecgs = [(filename, read_ISHNE(filename)) for filename in filenames]
    result = []
    wall = 0.0
    for filename, ecg in ecgs:
        with timer.Timer() as compression_time:
            compressed_leads = compress_leads(*ecg.leads)
        print "Compression:", compression_time
        wall += compression_time.interval
        with timer.Timer() as compute:
            y, x = get_hr(compressed_leads, ecg.sampling_rate / 4)
            result.append((x, y, filename,))
        wall += compute.interval
    print "Total:", wall, "seconds"
    for x, y, filename in result:
        plt.plot(x, y, label=filename)
    plt.legend()
    plt.title("ECG - RR")
    plt.xlabel("Hours")
    plt.ylabel("Heartrate (BPM)")
    plt.show()

def compress(leads, sampling_rate, filename, out_queue):
    with timer.Timer() as compression_time:
        compressed_leads = compress_leads(*leads)
    out_queue.put((compressed_leads, sampling_rate / 4, filename))
    print "Compression:", compression_time

def compute(in_queue, out_queue):
    load_CUDA()
    while True:
        work = in_queue.get()
        # To terminate the compute process, put None into its input Queue
        if work is True:
            # Terminate consumer
            out_queue.put(True)
            return

        with timer.GPUTimer(cuda) as compute:
            compressed_leads, sampling_rate, filename = work
            heartrate = get_hr(compressed_leads, sampling_rate)
        print "GPU (Transfer + Compute):", compute
        out_queue.put((heartrate, filename,))

def plot(in_queue):
    while True:
        work = in_queue.get()
        # To terminate the plot process, put None into input Queue
        if work is True:
            plt.title("ECG - RR")
            plt.xlabel("Hours")
            plt.ylabel("Heartrate (BPM)")
            plt.legend()
            plt.show()
            return
        heartrate, filename = work
        rr, indexes = heartrate
        plt.plot(indexes, rr, label=os.path.basename(filename))

def plot_hr_pipelined(ecg_filenames):
    manager = multiprocessing.Manager()
    compress_queue = manager.Queue()
    compute_queue = manager.Queue()
    compress_pool = multiprocessing.Pool(processes = 8)
    compute_process = multiprocessing.Process(target=compute, args=(compress_queue, compute_queue,))
    plot_process = multiprocessing.Process(target=plot, args=(compute_queue,))
    compute_process.start()
    plot_process.start()
    ecgs = [(filename, read_ISHNE(filename)) for filename in ecg_filenames]
    with timer.Timer() as wall:
        for filename, ecg in ecgs:
            compress_pool.apply_async(compress, args=(ecg.leads, ecg.sampling_rate, filename, compress_queue,))
        # Prevent more work from being put to the pool
        compress_pool.close()
        # # Wait for the pool to finish
        compress_pool.join()
        # Send the done message
        compress_queue.put(True)
        compute_process.join()
    # Total seems to include about 200ms of overhead
    print "Total:", wall
    plot_process.join()

def main():
    parser = argparse.ArgumentParser(description="plot ECG data")
    parser.add_argument("ecg", type=str, nargs="+", help="ECG file to process")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("-L", dest="leads", metavar="LEAD", nargs="+",
                            help="number of leads to plot", type=int)
    plot_group.add_argument("-HR", dest="plot_heartrate",
                            action="store_true", default=False,
                            help="plot RR data")
    args = parser.parse_args()
    if args.plot_heartrate:
        plot_hr_many(args.ecg)
    else:
        plot_leads(args.ecg, args.leads)

if __name__ == '__main__':
    main()
