#include "mandelbrot.h"
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

mandelbrot::mandelbrot(int H, int W, mandelbrot::Point center, double zoom, int max_iter): 
	height(H), width(W), 
	center(center), zoom(zoom), max_iter(max_iter), 
	//cr(mandelbrot::Array(height,width)),
	//ci(mandelbrot::Array(height,width)),
	//zr(mandelbrot::Array(height,width)),
	//zi(mandelbrot::Array(height,width)),
	values(mandelbrot::Array(height,width)),
	image(mandelbrot::ArrayCV(height,width,CV_8UC1)),
  GPU_object(applyIterGPU(height,width,max_iter))
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
  GPU_object.setMaxIter(max_iter);
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
  GPU_object.SET_COORD_VALS(center.real(),center.imag(),zoom);	
}

void mandelbrot::calcValues(){
	
	GPU_object.GPU_PAR_FOR();
  double* values_pointer = values.data();
  GPU_object.copyValues(values_pointer);
  
	//smoothColor();
	values = (values.array()==max_iter).select(0,values);
	
	double period_per_iter = 510./5000;
	double periods = 20;
	double K = periods*period_per_iter;
	values = ((K*values).array()-510*(K*values/510).array().floor());
	values = (values.array()<=255).select(values,(510-values.array()));
	values = values.array().round();	
}

/*void mandelbrot::smoothColor(){
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
}	*/
