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
											  values(Array<double,Dynamic,Dynamic>(height,width))
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
	Array<bool, Dynamic, Dynamic> valid_z(height, width);
	Array<double, Dynamic, Dynamic> zr2 = zr, zi2 = zi;
	
	while((zr2+zi2<=1e6).any() && (values<max_iter).all()){
		valid_z = ((zr2+zi2)<=1e6);
		values = (valid_z).select(values+1,values);
		zi = (valid_z).select(zi*zr,zi);
		zi = (valid_z).select(zi+zi+ci,zi);
		zr = (valid_z).select(zr2-zi2+cr,zr);
		zr2 = (valid_z).select(zr*zr,zr2);
		zi2 = (valid_z).select(zi*zi,zi2);
		
	}
	
	values = (values<max_iter).select(values,0);
	smoothColor();
	histColor();	
	
}

void mandelbrot::smoothColor(){
	Array <bool, Dynamic, Dynamic> valid_z = values!=0;
	Array <double, Dynamic, Dynamic> z2, log_z, nu;
	
	z2 = zr*zr + zi*zi;
	log_z = log(z2)/2;
	nu = log(log_z/log(2))/log(2);
	values = valid_z.select(values+ 1 - nu,0);
}

void mandelbrot::histColor(){
	Array<double,Dynamic,Dynamic> histogram = Array<double,Dynamic,Dynamic>::Zero(height, width);
	
	double m;
	for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
			m = values(j,i);
			if (m<max_iter)
				histogram(floor(m)) += 1;
		}
	}

	double total = histogram.sum();
	vector<double> hues;
	double h = 0;
	for (int i=0; i<max_iter; i++){
		h += histogram(i)/total;
		hues.push_back(h);
	}
	hues.push_back(h);

	for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
			m = values(j,i);
			double lerp = hues[floor(m)]*(1-fmod(m,1)) + hues[ceil(m)]*fmod(m,1);
			int hue = int(255*lerp);
			if (m<max_iter)
				values(j,i) = hue;
		}
	}
}	

/*void mandelbrot::histColor(){
	Array <double, 1, Dynamic> pixelCountPerIter = Array<double, 1, Dynamic>::Zero(max_iter);
	
	for (int i=0; i<width; i++){
		for (int j=0; j<height; j++){
			pixelCountPerIter(values(j,i))++;
		}
	}

	double total_counts = pixelCountPerIter.sum();

	Array <double, Dynamic, Dynamic> hues = Array<double, Dynamic, Dynamic>::Zero(height,width);
	
	for (int i=0; i<width; i++){
		for (int j=0; j<height; j++){
			double iters = values(j,i);
			for (int iter=1; iter<=iters; iter++){
					hues(j,i) += pixelCountPerIter(iter);
			}
		}
	}	
	hues = hues * 255/hues.maxCoeff();
	values = hues;

	Array<double,Dynamic,Dynamic> color1 = floor(values);
        Array<double,Dynamic,Dynamic> color2 = ceil(values);
        Array<double,Dynamic,Dynamic> t = values;

        for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
                        t(j,i) = fmod(values(j,i),1);
                }
        }

	values = color1*(1-t) + color2*t;
        values = values.round();
}*/	

void mandelbrot::createImage(string fname, bool disp, bool save){
	cv::Mat values_cv(height, width, CV_8UC1);
	cv::transpose(octane::eigen2cv(values.template cast<uint8_t>()),values_cv);
	
	cv::Mat image;
	cv::applyColorMap(values_cv, image, 5);
	
	if (save){
		string save_loc = "../images/";
		imwrite(save_loc + fname + ".jpg",image);
	}

	if (disp){
		cv::namedWindow(fname, CV_WINDOW_NORMAL);
		cv::setWindowProperty(fname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
		cv::imshow(fname,image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return;
}

