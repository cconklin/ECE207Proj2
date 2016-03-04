custom_functions.o: custom_functions.cu
	nvcc $^ -o $@ -lm -shared -O3
