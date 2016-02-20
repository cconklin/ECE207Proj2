from ctypes import *
from matplotlib import pyplot as plt
import numpy
num = 64
computedb = CDLL("computedb.o")
wavelet = numpy.zeros(num).astype(numpy.float32)
wavelet_p = wavelet.ctypes.data_as(POINTER(c_float))
num_points = c_int(num)
computedb.generate_wavelet(wavelet_p, num_points)
if __name__ == '__main__':    
    x = numpy.linspace(0, (num - 1), num=num)
    plt.plot(x, wavelet)
    plt.show()
