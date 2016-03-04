import ctypes
import numpy
import os
dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_functions.o")
custom_functions = ctypes.CDLL(dll_path)

def compress_ecg(lead1, lead2, lead3, threshold=0.5):
    lead_len = len(lead1)
    output1 = numpy.zeros(lead_len).astype(numpy.float32)
    output2 = numpy.zeros(lead_len).astype(numpy.float32)
    output3 = numpy.zeros(lead_len).astype(numpy.float32)
    samples = numpy.zeros(lead_len).astype(numpy.float32)
    input1_p = lead1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output1_p = output1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input2_p = lead2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output2_p = output2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input3_p = lead3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output3_p = output3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    samples_p = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    threshold = ctypes.c_float(threshold)
    input_len = ctypes.c_int(lead_len)
    output_len = ctypes.c_int(0)
    output_len_p = ctypes.pointer(output_len)
    custom_functions.threshold_ecg(output1_p, output2_p, output3_p,
                                   samples_p, output_len_p, input1_p,
                                   input2_p, input3_p, input_len, threshold)
    output1 = output1[:output_len.value]
    output2 = output2[:output_len.value]
    output3 = output3[:output_len.value]
    samples = samples[:output_len.value]
    return samples, output1, output2, output3, output_len.value

def turning_point_compression(lead, times=1):
    lead_len = len(lead)
    output = numpy.zeros(lead_len / 2).astype(numpy.float32)
    inputP = lead.astype(numpy.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    outputP = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    for i in range(times):
        input_len = ctypes.c_int(lead_len)
        custom_functions.turning_point_compress(outputP, inputP, input_len)
        inputP = outputP
        lead_len = lead_len / 2
    return output[:lead_len]

def inclusive_scan(out_device_ary, in_device_ary, length):
    in_p = ctypes.cast(int(in_device_ary), ctypes.POINTER(ctypes.c_int))
    out_p = ctypes.cast(int(out_device_ary), ctypes.POINTER(ctypes.c_int))
    c_length = ctypes.c_int(length)
    custom_functions.inclusive_scan(out_p, in_p, c_length)

def exclusive_scan(out_device_ary, in_device_ary, length):
    in_p = ctypes.cast(int(in_device_ary), ctypes.POINTER(ctypes.c_int))
    out_p = ctypes.cast(int(out_device_ary), ctypes.POINTER(ctypes.c_int))
    c_length = ctypes.c_int(length)
    custom_functions.exclusive_scan(out_p, in_p, c_length)
