custom_functions.o: custom_functions.cu
	nvcc $^ -o $@ -lm -shared -O3 -lpthread

plotecg.o: plotecg.cu kernels.cu
	nvcc $^ -o $@ -lpthread -O3 -lm -shared
