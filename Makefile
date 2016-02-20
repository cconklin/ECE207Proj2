computedb.o: computedb.c
	gcc -o $@ $? --shared -lm