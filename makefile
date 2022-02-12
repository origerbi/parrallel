all: program

program: cuda.o main.o
	g++ main.o cuda.o -L/usr/local/cuda/lib64 -lcuda -lcudart -fopenmp -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -o program

cuda.o:
	nvcc -c cuda.cu -o cuda.o

main.o:
	mpicc -c main.c -o main.o

clean: 
	rm -f *.o program