#include "mandelbrot.h"
#include "Eigen2CV.h"
#include "tbb/tbb.h"
#include "applyIter.h"
#include <cmath>
#include <vector>

// hello
using namespace octane;

mandelbrot::mandelbrot(int H, int W, mandelbrot::Point center, double zoom, size_t max_iter): 
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
		cv::transpose(eigen2cv(values.template cast<uint8_t>()),image);
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
	
	parallel_for(blocked_range2d<size_t>(0, height, 0, width), applyIter(values,zr,zi,cr,ci,max_iter));
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
	
	smoothColor();
	values = (values==max_iter).select(0,values);
	//histColor();
	
	double K = 510*20/5000;
	values = ((K*values)-510*(K*values/510).floor());
	values = (values<=255).select(values,(510-values));
	
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
				log_z = std::log(z2)/2;
				nu = std::log(log_z/std::log(2))/std::log(2);
				values(j,i) = values(j,i) + 1 - nu;
			}
		}
	}
}

void mandelbrot::histColor(){
	mandelbrot::Array histogram = mandelbrot::Array::Zero(max_iter, 1);
	
	double m;
	for (int i=0; i<width; i++){
                for (int j=0; j<height; j++){
			m = values(j,i);
			if (m<max_iter)
				histogram(floor(m)) += 1;
		}
	}

	double total = histogram.sum();
	std::vector<double> hues;
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
