custom_functions.o: custom_functions.c
	gcc $^ -o $@ -lm -shared -fPIC -O3
