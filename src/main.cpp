#include "mandelbrot.h"
#include <complex>
#include <string>
#include <iostream>
#include <cmath>
#include <time.h>

using namespace std;

int main(int argc, char *argv[]){

	cout << argv[1] << endl;

	int height,width;
	complex<double> center;
	double x,y,zoom;
	uint64_t max_iter;

	cin >> height >> width;
	cin >> x >> y;
	cin >> zoom >> max_iter;

	center= complex<double>(x,y);
	
	
	if (strcmp(argv[1],"0") == 0){
		mandelbrot a(height,width,center,zoom,max_iter);
		a.calcValues();
		a.createImage("hello");
	}
	
	else if(strcmp(argv[1],"1") == 0){
		for (int i=0; i<960*2; i++){
			string fname = "image" + to_string(i);
			zoom*=pow(10,.00625);
			mandelbrot a(height,width,center,zoom,max_iter);
			a.calcValues();
			a.createImage(fname,false,true);
		}
	}
	
	return 0;
}
	
