#include "mandelbrot.h"
#include <complex>
#include <iostream>

int main(){

	int height,width;
	complex<double> center;
	double x,y,zoom;
	uint64_t max_iter;

	cin >> height >> width;
	cin >> x >> y;
	cin >> zoom >> max_iter;

	center= complex<double>(x,y);

	//mandelbrot a(height,width,-0.761-1e-3,-0.761+1e-3,-0.0841-1e-3,-0.0841+1e-3);
	mandelbrot a(height,width,center,zoom,max_iter);
	a.calcValues();
	a.createImage("hello");
	
	return 0;
}
	
