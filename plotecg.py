#!/usr/bin/env python

# Designed for Python 2 (will NOT run on Python 3)

import argparse
import pycuda.autoinit
import pycuda.driver as cuda
import numpy
import matplotlib.pyplot as plt
import ishne
from pycuda.compiler import SourceModule
import sys
import timer
import custom_functions
timer.driver = cuda

with open("plotecg.cu") as wavelet_file:
    mod = SourceModule(wavelet_file.read())

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

runtime = 0.0

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
    global runtime

    with timer.Timer() as tot:
        with timer.Timer() as calc:
            # Kernel Parameters
            threads_per_block = 200
            num_blocks = lead_size / threads_per_block
        if verbose:
            print "Kernel Size Calculation:", calc.interval

        with timer.Timer() as corr:

            # correlate lead with wavelet
            correlated = cuda.mem_alloc(lead_size * 4)
            cross_correlate_with_wavelet(correlated, d_lead, d_wavelet,
                                         numpy.int32(lead_size),
                                         numpy.int32(wavelet_len),
                                         grid=(num_blocks, 1),
                                         block=(threads_per_block, 1, 1))
            cuda.Context.synchronize()

        with timer.Timer() as thresh:
            # threshold correlated lead
            thresholded_signal = cuda.mem_alloc(lead_size * 4)
            threshold(thresholded_signal, correlated,
                      numpy.float32(threshold_value),
                      grid=(num_blocks, 1), block=(threads_per_block, 1, 1))
            cuda.Context.synchronize()

        if verbose:
            print "Correlate:", corr.interval
            print "Threshold:", thresh.interval

    runtime += tot.interval
    if verbose:
        print "Preprocess Lead:", tot.interval

    return thresholded_signal

def preprocess(d_lead1, d_lead2, d_lead3, lead_size,
               wavelet, threshold_value):
    with timer.Timer() as wav:
        d_wavelet = cuda.to_device(wavelet)
        wavelet_len = len(wavelet)
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
                                                    lead_size)
    global runtime
    runtime += wav.interval
    if verbose:
        print "Wavelet Transfer:", wav.interval
    return (d_merged_lead, lead_len)

def synchronize_and_merge(d_tlead1, d_tlead2, d_tlead3, length):
    with timer.Timer() as sm:
        # synchronize
        (offset1, offset2, offset3, lead_len) = synchronize(d_tlead1,
                                                            d_tlead2,
                                                            d_tlead3,
                                                            length)
        # merge
        d_merged_lead, lead_len = merge(d_tlead1, offset1, d_tlead2, offset2,
                                        d_tlead3, offset3, lead_len)
        cuda.Context.synchronize()

    global runtime
    runtime += sm.interval
    if verbose:
        print "Synchronize & Merge:", sm.interval

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

def synchronize(d_tlead1, d_tlead2, d_tlead3, length):
    # Number of points to use to synchronize
    chunk = ecg.sampling_rate * 2
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

def get_heartbeat(d_lead, length):
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
                   numpy.int32(ecg.sampling_rate / 4),
                   numpy.int32(c_length),
                   grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    clean_result(dev_rr, numpy.int32(120), numpy.int32(40),
                 numpy.int32(1), numpy.int32(c_length),
                 grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    moving_average_filter(dev_rr, c_length, 250)

    index = cuda.from_device(cd_index, (c_length,), numpy.int32)
    rr = cuda.from_device(dev_rr, (c_length,), numpy.int32)
    index[0] = index[1]

    return rr, index / float(ecg.sampling_rate * 3600)

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
    global ecg
    ecg = ishne.ISHNE(ecg_filename)
    ecg.read()
    

def plot_leads(ecg_filename, lead_numbers):

    read_ISHNE(ecg_filename)
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

def plot_hr(ecg_filename):

    read_ISHNE(ecg_filename)

    # number of samples: 0.06 - 0.1 * SAMPLING_RATE (QRS Time: 60-100ms)
    num_samples = int(0.08 * ecg.sampling_rate / 4) + 2

    hat = generate_hat(num_samples)

    with timer.Timer() as compression:
        c_leads = compress_leads(*ecg.leads)
    if verbose:
        print "Compress:", compression.interval

    with timer.Timer() as transfer:
        d_lead1, d_lead2, d_lead3, length = transfer_leads(*c_leads)
    if verbose:
        print "Transfer:", transfer.interval

    with timer.Timer() as pre_time:
        d_mlead_hat, length_hat = preprocess(d_lead1, d_lead2, d_lead3,
                                             length, hat, 0.5)

    with timer.Timer() as time:
        y, x = get_heartbeat(d_mlead_hat, length_hat)

    if verbose:
        print "Hat Preprocess:", pre_time.interval
        print "RR time:", time.interval

    print "HR processed in", pre_time.interval + time.interval + \
          transfer.interval + compression.interval, "seconds"
    if verbose:
        print "\nTime Breakdown:"
        print "\tPython Overhead:", pre_time.interval - runtime, "seconds"
        print "\t\t(Mostly function calls to helper functions)"
        print "\tTransfer Time:", transfer.interval, "seconds"
        print "\t\t(Transfer 3 leads to GPU)"
        print "\tHR and Post Processing:", time.interval, "seconds"
        print "\t\t(Edge Detection, Distance Between Peaks, Post Filering, "\
              "Transfer from GPU)"
        print "\tLead Processing:", runtime, "seconds"
        print "\t\t(Cross Correlation, Thresholding, Synchronization & Merging"

    cuda.Context.synchronize()
    plt.figure(1)
    plt.plot(x, y)
    plt.title("ECG - RR")
    plt.xlabel("Hours")
    plt.ylabel("Heartrate (BPM)")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="plot ECG data")
    parser.add_argument("ecg", type=str, help="ECG file to process")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("-L", dest="leads", metavar="LEAD", nargs="+",
                            help="number of leads to plot", type=int)
    plot_group.add_argument("-HR", dest="plot_heartrate",
                            action="store_true", default=False,
                            help="plot RR data")
    parser.add_argument("--verbose", help="show all times",
                        default=False, action="store_true")
    args = parser.parse_args()
    global verbose
    verbose = args.verbose
    if args.plot_heartrate:
        plot_hr(args.ecg)
    else:
        plot_leads(args.ecg, args.leads)

if __name__ == '__main__':
    main()
