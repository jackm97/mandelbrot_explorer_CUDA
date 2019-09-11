all: main.o mandelbrot.o
	g++ main.o mandelbrot.o -o mandelbrot_explorer -ltbb -lrt `pkg-config --cflags --libs opencv`

main.o: main.cpp mandelbrot.h applyIter.h Eigen2CV.h
	g++ -I /usr/include/eigen3 -fopenmp -march=native -msse2 -O2 -c main.cpp -o main.o `pkg-config --cflags --libs opencv`

mandelbrot.o: mandelbrot.cpp mandelbrot.h applyIter.h Eigen2CV.h
	g++ -I /usr/include/eigen3 -fopenmp -march=native -msse2 -O2 -c mandelbrot.cpp -o mandelbrot.o `pkg-config --cflags --libs opencv`
