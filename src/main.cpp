#include "mandelbrot.h"
#include <complex>
#include <string>
#include <iostream>
#include <cmath>

using namespace std;

int main(){

	int height,width;
	complex<double> center;
	double x,y,zoom;
	uint64_t max_iter;

	cin >> height >> width;
	cin >> x >> y;
	cin >> zoom >> max_iter;

	center= complex<double>(x,y);

	mandelbrot a(height,width,center,zoom,max_iter);
	a.calcValues();
	a.createImage("hello");
	
	/*for (int i=0; i<960; i++){
		string fname = "image" + to_string(i);
		zoom*=pow(10,.00625*2);
		mandelbrot a(height,width,center,zoom,max_iter);
		a.calcValues();
		a.createImage(fname,false,true);
	}*/
	
	return 0;
}
	
