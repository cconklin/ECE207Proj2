custom_functions.o: custom_functions.cu
	nvcc $^ -o $@ -O3 -lm -lpthread -shared -Xcompiler -fPIC

plotecg.o: plotecg.cu kernels.cu
	nvcc $^ -o $@ -lpthread -O3 -lm -shared
