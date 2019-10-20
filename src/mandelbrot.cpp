#include "mandelbrot.h"
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "tbb/tbb.h"
#include "applyIter.h"

mandelbrot::mandelbrot(int H, int W, mandelbrot::Point center, double zoom, int max_iter): 
	height(H), width(W), 
	center(center), zoom(zoom), max_iter(max_iter), 
	cr(mandelbrot::Array(height,width)),
	ci(mandelbrot::Array(height,width)),
	zr(mandelbrot::Array(height,width)),
	zi(mandelbrot::Array(height,width)),
	values(mandelbrot::Array(height,width)),
	image(mandelbrot::ArrayCV(height,width,CV_8UC1))
{
	resetValues();
}

void mandelbrot::changeCenter(mandelbrot::Point new_center){
	center = new_center;
	isCalc = false;
}

void mandelbrot::changeZoom(double new_zoom){
	zoom = new_zoom;
	isCalc = false;
}

void mandelbrot::changeMaxIter(size_t new_max_iter){
	max_iter = new_max_iter;
	isCalc = false;
}

mandelbrot::ArrayCV mandelbrot::getImageCV(){
	if (!isCalc){
		resetValues();
		calcValues();
		cv::eigen2cv(values,image);
		image.convertTo(image,CV_8UC1);
		isCalc=true;
	}

	return image;
}

void mandelbrot::resetValues(){	
	double aspect_ratio = double(width)/height;
	double x_range, y_range, xmin, ymin, intervalx, intervaly;
	if (aspect_ratio<1){
		x_range = 4/zoom;
		y_range = (1/aspect_ratio)*4/zoom;
	}
	else{
		x_range = (aspect_ratio)*4/zoom;
		y_range = 4/zoom;
	}
	xmin = center.real() - x_range/2;
	ymin = center.imag() - y_range/2;
	intervalx = x_range/width;
	intervaly = y_range/height;

	for (int i=0; i<width; i++){
		for (int j=0; j<height; j++){
			mandelbrot::Point point(xmin + i*intervalx, ymin + j*intervaly);
			cr(j,i) = point.real();
			ci(j,i) = point.imag();
			zr(j,i) = 0;
			zi(j,i) = 0;
			values(j,i) = 0;
		}
	}
}

void mandelbrot::calcValues(){
	
	using namespace tbb;
	applyIter parallel_object = applyIter(values,zr,zi,cr,ci,max_iter);
	parallel_for(blocked_range2d<size_t>(0, height, 0, width), applyIter(values,zr,zi,cr,ci,max_iter));
	smoothColor();
	values = (values.array()==max_iter).select(0,values);
	
	double period = 510;
	double period_per_iter = 20/5000;
	double K = period*period_per_iter;
	values = ((K*values).array()-510*(K*values/510).array().floor());
	values = (values.array()<=255).select(values,(510-values.array()));
	values = values.array().round();	
}

void mandelbrot::smoothColor(){
	double z2, log_z, nu;

	for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
			if (values(j,i)!=max_iter){	
				z2 = zr(j,i) * zr(j,i) + zi(j,i) * zi(j,i);
				log_z = std::log(z2)/2;
				nu = std::log(log_z/std::log(2))/std::log(2);
				values(j,i) = values(j,i) + 1 - nu;
			}
		}
	}
}	
