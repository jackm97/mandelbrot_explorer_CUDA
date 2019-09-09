#define NDEBUG

#include "mandelbrot.h"
#include "Eigen2CV.h"
#include "tbb/tbb.h"
#include "applyIter.h"
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

mandelbrot::mandelbrot(int H, int W, complex<double> center, double zoom, size_t max_iter): height(2*H), width(2*W), 
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

	tbb::parallel_for(blocked_range2d<size_t>(0, height, 0, width), applyIter(values,zr,zi,cr,ci,max_iter));
	/*double iters=0,
	       R2=1e6,
	       zr2=0,
	       zi2=0;
	
	for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
			iters=0;
			zr2=0;
			zi2=0;
			while((zr2+zi2<=R2) && (iters<max_iter)){
				zi(j,i) = zi(j,i) * zr(j,i);
				zi(j,i) = zi(j,i) + zi(j,i) + ci(j,i);
				zr(j,i) = zr2 - zi2 + cr(j,i);
				zr2 = zr(j,i) * zr(j,i);
				zi2 = zi(j,i) * zi(j,i);
				iters++;
			}
			values(j,i) = iters;
		}
	}*/
	
	/*Array <double, Dynamic, Dynamic> logz = log(log(zr2+zi2)/power);	
	double K = logz.maxCoeff() - logz.minCoeff();
	values = (values<max_iter).select(logz/K,max_iter);
	values = (values!=max_iter).select(255*sin(2*M_PI*values),0);
	values = round(values);*/

	smoothColor();
	values = (values==max_iter).select(0,values);
	//histColor();
	
	double K = 4;
	values = ((K*values/510)-(K*values/510).floor());
	values = (values<.5).select(255*values,255*(1-values));
	//values = 255/2*(1 - cos(2*M_PI*values/K));
	//values = 1 - (1-values/values.maxCoeff()).pow(1./6);

	/*double min,max;
	min = values.minCoeff();
	max = values.maxCoeff();
	values = (values!=0).select(255*(values-min)/(max-min),0);*/
	values = values.round();	
}

void mandelbrot::smoothColor(){
	double z2, log_z, nu;

	for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
			if (values(j,i)!=max_iter){	
				z2 = zr(j,i) * zr(j,i) + zi(j,i) * zi(j,i);
				log_z = log(z2)/2;
				nu = log(log_z/log(2))/log(2);
				values(j,i) = values(j,i) + 1 - nu;
			}
		}
	}
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
			uint64_t hue = uint64_t(max_iter*lerp);
			if (m<max_iter)
				values(j,i) = hue;
			else 
				values(j,i) = 0;
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
	cv::applyColorMap(values_cv, image, cv::COLORMAP_BONE);
	cv::resize(image,image,cv::Size(),.5,.5);
	
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

