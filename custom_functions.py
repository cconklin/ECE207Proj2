import ctypes
import numpy
custom_functions = ctypes.CDLL("custom_functions.o")

def compress_ecg(lead1, lead2, lead3, threshold=0.5):
    lead_len = len(lead1)
    output1 = numpy.zeros(size).astype(numpy.float32)
    output2 = numpy.zeros(size).astype(numpy.float32)
    output3 = numpy.zeros(size).astype(numpy.float32)
    samples = numpy.zeros(size).astype(numpy.float32)
    input1_p = lead1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output1_p = output1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input2_p = lead2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output2_p = output2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input3_p = lead3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output3_p = output3.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    samples_p = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    threshold = ctypes.c_float(threshold)
    input_len = ctypes.c_int(size)
    output_len = ctypes.c_int(0)
    output_len_p = ctypes.pointer(output_len)
    custom_functions.threshold_ecg(output1_p, output2_p, output3_p, samples_p, output_len_p, input1_p, input2_p, input3_p, input_len, threshold)
    output1 = output1[:output_len.value]
    output2 = output2[:output_len.value]
    output3 = output3[:output_len.value]
    samples = samples[:output_len.value]
    return output1, output2, output3, samples, output_len.value
