CC = g++
CCFLAGS = -fopenmp -march=native -msse2 -pthread -O2
LDFLAGS = `pkg-config --cflags --libs opencv4`
INC = -I /usr/include/eigen3
LIB = -ltbb -lrt



all: main.o mandelbrot.o
	$(CC) main.o mandelbrot.o -o mandelbrot_explorer $(LIB) $(LDFLAGS)

main.o: main.cpp mandelbrot.h
	$(CC) $(CCFLAGS) $(INC) -c main.cpp -o main.o $(LDFLAGS) 

mandelbrot.o: mandelbrot.cpp mandelbrot.h applyIter.h Eigen2CV.h
	$(CC) $(CCFLAGS) $(INC) -c mandelbrot.cpp -o mandelbrot.o $(LDFLAGS)

clean:
	rm -rf *.o mandelbrot_explorer
