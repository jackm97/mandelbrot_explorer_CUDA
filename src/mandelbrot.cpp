#define NDEBUG

#include "mandelbrot.h"
#include "Eigen2CV.h"
#include <complex>
#include <Eigen/Dense>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <vector>


using namespace std;
using Eigen::Array;
using Eigen::Dynamic;

mandelbrot::mandelbrot(int H, int W, complex<double> center, double zoom, uint64_t max_iter): height(H), width(W), 
											  cr(Array<double,Dynamic,Dynamic>(height,width)),
											  ci(Array<double,Dynamic,Dynamic>(height,width)),
											  zr(Array<double,Dynamic,Dynamic>(height,width)),
											  zi(Array<double,Dynamic,Dynamic>(height,width)),
											  values(Array<uint64_t,Dynamic,Dynamic>(height,width))
{	
	this->max_iter = max_iter;
	double aspect_ratio = double(W)/H;
	double x_range, y_range, xmin, ymin, intervalx, intervaly;
	if (aspect_ratio<1){
		x_range = 4/zoom;
		y_range = (1/aspect_ratio)*4/zoom;
	}
	else{
		x_range = (aspect_ratio)*4/zoom;
		y_range = 4/zoom;
	}
	xmin = real(center) - x_range/2;
	ymin = imag(center) - y_range/2;
	intervalx = x_range/width;
	intervaly = y_range/height;

	for (int i=0; i<width; i++){
		for (int j=0; j<height; j++){
			complex<double> point(xmin + i*intervalx, ymin + j*intervaly);
			cr(j,i) = real(point);
			ci(j,i) = imag(point);
			zr(j,i) = 0;
			zi(j,i) = 0;
			values(j,i) = 0;
		}
	}
}

void mandelbrot::calcValues(){
	Array<uint64_t, Dynamic, Dynamic> valid_z(height, width);
	Array<double, Dynamic, Dynamic> zr2 = zr, zi2 = zi;

	while(((zr2+zi2)<=4 && values<=max_iter).any()){
		valid_z = ((zr2+zi2)<4).template cast<uint64_t>();
		zi = zi*zr;
		zi += zi;
		zi += ci;
		zr = zr2 - zi2 + cr;
		zr2 = zr*zr;
		zi2 = zi*zi;
		values =  valid_z + values;
	}

	values = values;
	values = values * ((values<max_iter).template cast<uint64_t>());
	values = values*255/(max_iter-1);	
	
}

void mandelbrot::createImage(string fname){
	cv::Mat values_cv(height, width, CV_8UC1);
	cv::transpose(octane::eigen2cv(values.template cast<uint8_t>()),values_cv);

	cv::Mat image;
	cv::applyColorMap(values_cv, image, 5);
	
	cv::namedWindow(fname, CV_WINDOW_NORMAL);
	cv::setWindowProperty(fname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	cv::imshow(fname,image);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return;
}

