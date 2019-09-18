CC = g++
CCFLAGS = -fopenmp -march=native -msse2 -O2
LDFLAGS = `pkg-config --cflags --libs opencv`
INC = -I /usr/include/eigen3
LIB = -ltbb -lrt



all: main.o mandelbrot.o
	$(CC) main.o mandelbrot.o -o mandelbrot_explorer $(LIB) $(LDFLAGS)

main.o: main.cpp mandelbrot.h
	$(CC) $(CCFLAGS) $(INC) -c main.cpp -o main.o 

mandelbrot.o: mandelbrot.cpp mandelbrot.h applyIter.h Eigen2CV.h
	$(CC) $(CCFLAGS) $(INC) -c mandelbrot.cpp -o mandelbrot.o 

clean:
	rm -rf *.o mandelbrot_explorer
