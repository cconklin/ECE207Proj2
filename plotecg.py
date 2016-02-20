#!/usr/bin/env python

# Designed for Python 2 (will NOT run on Python 3)

import argparse
import pycuda.autoinit
import pycuda.driver as cuda
import numpy
import scipy.signal
import matplotlib.pyplot as plt
import ishne
from pycuda.compiler import SourceModule
import sys
from timer import Timer
import computedb

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

runtime = 0.0

def transfer_leads(*h_leads):
    length = len(h_leads[0])
    return tuple(cuda.to_device(h_lead.astype(numpy.float32))
                 for h_lead in h_leads) + (length,)

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

def generate_qrs_wavelet(lead, seed_wavelet):
    #SJM
    start_clip=0
    lead_clip = numpy.asarray(lead[start_clip:start_clip+2000],numpy.float32)
    lead_clip_size=len(lead_clip)

    # Kernel Parameters
    threads_per_block = 200
    num_blocks = lead_clip_size / threads_per_block

    correlated_clip = numpy.zeros(lead_clip_size).astype(numpy.float32)
    thresholded_clip_signal = numpy.zeros(lead_clip_size).astype(numpy.int32)
    edge_clip_signal=numpy.zeros(lead_clip_size).astype(numpy.int32)

    cross_correlate_with_wavelet(cuda.Out(correlated_clip),
                                 cuda.In(lead_clip),
                                 cuda.In(seed_wavelet),
                                 numpy.int32(lead_clip_size),
                                 numpy.int32(len(seed_wavelet)),
                                 grid=(num_blocks, 1),
                                 block=(threads_per_block, 1, 1))

    threshold(cuda.Out(thresholded_clip_signal),
              cuda.In(correlated_clip),
              numpy.float32(1.0),
              grid=(num_blocks, 1),
              block=(threads_per_block, 1, 1))

    edge_detect(cuda.Out(edge_clip_signal),
                cuda.In(thresholded_clip_signal),
                grid=(num_blocks, 1),
                block=(threads_per_block, 1, 1))

    first_r_peak=0
    for x in xrange(2000):
       if(edge_clip_signal[x]==1):
           first_r_peak=x
           break

    # This should be sensitive to the sampling rate
    qrs = lead[first_r_peak-6:first_r_peak+10]
    auto_pick_qrs_wavelet = numpy.asarray(qrs, numpy.float32)
    return auto_pick_qrs_wavelet

def median_filter(out_array, in_ary, grid, block):
    padded = numpy.pad(in_ary, (1, 1), mode="edge")
    filter(cuda.Out(out_array), cuda.In(padded), grid=grid, block=block)
    return out_array

# Note: Inlining this saves 50ms per invocation
def preprocess_lead(d_lead, lead_size, d_wavelet,
                    wavelet_len, threshold_value):
    global runtime

    with Timer() as tot:
        with Timer() as calc:
            # Kernel Parameters
            threads_per_block = 200
            num_blocks = lead_size / threads_per_block
        if verbose:
            print "Kernel Size Calculation:", calc.interval

        with Timer() as corr:

            # correlate lead with wavelet
            correlated = cuda.mem_alloc(lead_size * 4)
            cross_correlate_with_wavelet(correlated, d_lead, d_wavelet,
                                         numpy.int32(lead_size),
                                         numpy.int32(wavelet_len),
                                         grid=(num_blocks, 1),
                                         block=(threads_per_block, 1, 1))
            cuda.Context.synchronize()

        with Timer() as thresh:
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
    with Timer() as wav:
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
    with Timer() as sm:
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

    # Allocate output
    full_rr_signal = numpy.zeros(length / 64).astype(numpy.int32)

    # Get RR
    window_size = 8
    edge_signal = cuda.mem_alloc(4 * length)
    
    edge_detect(edge_signal, d_lead,
                grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    indecies = numpy.zeros(length / 64).astype(numpy.int32)
    masks = cuda.to_device(numpy.zeros(length / 64).astype(numpy.int32))
    d_index = cuda.to_device(indecies)
    index_of_peak(d_index, masks, edge_signal,
                  grid=(num_blocks, 1), block=(threads_per_block, 1, 1))

    get_rr(cuda.InOut(full_rr_signal), d_index, masks,
             numpy.int32(window_size), numpy.int32(ecg.sampling_rate),
             numpy.int32(len(indecies)),
             grid=(num_blocks / 64, 1), block=(threads_per_block, 1, 1))

    rr_signal = full_rr_signal[full_rr_signal != 0]

    # Filter

    # Reject the obvious outliers
    smoothed_rr_signal = rr_signal[rr_signal < 120]
    smoothed_rr_signal = smoothed_rr_signal[smoothed_rr_signal > 10]
    smoothed_rr_signal2 = numpy.copy(smoothed_rr_signal)

    # Median Reduce + Filter
    for i in range(3):
        if len(smoothed_rr_signal2) > 2187 * 3:
            median(cuda.Out(smoothed_rr_signal2),
                   cuda.In(numpy.copy(smoothed_rr_signal2)),
                   grid=(len(smoothed_rr_signal2) / 2187, 1),
                   block=(729, 1, 1))
        elif 1024 < len(smoothed_rr_signal2) <= 2187 * 3:
            median(cuda.Out(smoothed_rr_signal2),
                   cuda.In(numpy.copy(smoothed_rr_signal2)),
                   grid=(len(smoothed_rr_signal2) / 729, 1),
                   block=(81, 1, 1))
        else:
            median(cuda.Out(smoothed_rr_signal2),
                   cuda.In(numpy.copy(smoothed_rr_signal2)),
                   grid=(1, 1),
                   block=(len(smoothed_rr_signal2), 1, 1))
        # Since we just reduced the size of the array by a factor of 3,
        # we also need to reduce the size of the output vector
        smoothed_rr_signal2 = smoothed_rr_signal2[:len(smoothed_rr_signal2)/3]

        if len(smoothed_rr_signal2) > 2187 * 3:
            median_filter(smoothed_rr_signal2,
                          numpy.copy(smoothed_rr_signal2),
                          grid=(len(smoothed_rr_signal2) / 2187, 1),
                          block=(729, 1, 1))
        elif 1024 < len(smoothed_rr_signal2) <= 2187 * 3:
            median_filter(smoothed_rr_signal2,
                          numpy.copy(smoothed_rr_signal2),
                          grid=(len(smoothed_rr_signal2) / 729, 1),
                          block=(81, 1, 1))
        else:
            median_filter(smoothed_rr_signal2,
                          numpy.copy(smoothed_rr_signal2),
                          grid=(9, 1),
                          block=(len(smoothed_rr_signal2) / 9, 1, 1))

    # Use a better median filter for the last bit
    smoothed_rr_signal2 = scipy.signal.medfilt(smoothed_rr_signal2, (21,))

    return smoothed_rr_signal2

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
    num_samples = int(0.08 * ecg.sampling_rate)

    hat = generate_hat(num_samples)
    qrs_wavelet = generate_qrs_wavelet(ecg.leads[0], hat)

    with Timer() as transfer:
        d_lead1, d_lead2, d_lead3, length = transfer_leads(*ecg.leads)
    if verbose:
        print "Transfer:", transfer.interval

    with Timer() as pre_time:
        if verbose:
            print "Mexican Hat Wavelet:"
            print "--------------------"
        d_mlead_hat, length_hat = preprocess(d_lead1, d_lead2, d_lead3,
                                             length, hat, 1.0)
        if verbose:
            print "QRS Wavelet:"
            print "------------"
        d_mlead_qrs, length_qrs = preprocess(d_lead1, d_lead2, d_lead3,
                                             length, qrs_wavelet, 2.3)
        if verbose:
            print "DB4 Wavelet:"
            print "------------"
        d_mlead_db4, length_db4 = preprocess(d_lead1, d_lead2, d_lead3,
                                             length, computedb.wavelet, 3.5)

        length = min(length_hat, length_qrs, length_db4)

        if verbose:
            print "Combining Leads:"
            print "----------------"
        
        final_lead, final_length = synchronize_and_merge(d_mlead_hat,
                                                         d_mlead_qrs,
                                                         d_mlead_db4,
                                                         length)
    
    with Timer() as time:
        y = get_heartbeat(final_lead, final_length)

    if verbose:
        print "Total Preprocess:", pre_time.interval
        print "RR time:", time.interval

    print "HR processed in", pre_time.interval + time.interval + \
          transfer.interval, "seconds"
    print "\nTime Breakdown:"
    print "\tPython Overhead:", pre_time.interval - runtime, "seconds"
    print "\t\t(Mostly function calls to helper functions)"
    print "\tTransfer Time:", transfer.interval, "seconds"
    print "\t\t(Transfer 3 leads to GPU)"
    print "\tHR and Post Processing:", time.interval, "seconds"
    print "\t\t(Edge Detection, Distance Between Peaks, Post Filering, "\
          "Transfer from GPU)"
    print "\tLead Processing:", runtime, "seconds"
    print "\t\t(Cross Correlation, Thresholding, Synchronization & Merging"\
          " for 3 leads & 3 wavelets)"

    x = numpy.linspace(0, 23, num=len(y))
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
