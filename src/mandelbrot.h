#define NDEBUG

#include <Eigen/Dense>
#include <complex>
#include <opencv2/opencv.hpp>

#ifndef MANDELBROT_H
#define MANDELBROT_H

class mandelbrot{
	public:
		typedef std::complex<double> Point;
		typedef Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> Array;
		typedef cv::Mat ArrayCV; 
		
		mandelbrot(int H, int W, mandelbrot::Point center, double zoom, size_t max_iter);
		void changeCenter(mandelbrot::Point new_center);
		void changeZoom(double new_zoom);
		void changeMaxIter(size_t new_max_iter);
		mandelbrot::ArrayCV getImageCV();
	
	private:
		int height;
		int width;
		
		mandelbrot::Point center;
		double zoom;
		size_t max_iter;
		
		bool isCalc=false;
		
		mandelbrot::Array cr;
		mandelbrot::Array ci;
		mandelbrot::Array zr;
		mandelbrot::Array zi;
		
		mandelbrot::Array values;
		mandelbrot::ArrayCV image;

		void resetValues();
		void calcValues();
		void smoothColor();
		void histColor();
};

#endif
