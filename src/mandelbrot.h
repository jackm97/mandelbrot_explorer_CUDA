#include <string>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "applyIterGPU.h"

#ifndef MANDELBROT_H
#define MANDELBROT_H


// mandelbrot class used to render images of the Mandelbrot Set
// 	Example (generate 1080x1920 image):
//
// 	int height=1080, width=1920;
// 	mandelbrot::Point center(0,0);
// 	double zoom=1.;
// 	int max_iter = 5000;
// 	mandelbrot m(height, width, center, zoom, max_iter);
//
// 	mandelbrot::ArrayCV image = m.getImageCV();
//
class mandelbrot{
	public:
		typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic> Array;
		typedef cv::Mat ArrayCV; 
		
		// Constructor:
		// 	Inputs:
		// 	H - resolution height
		// 	W - resolution width
		// 	center - complex center of image (real,imag)
		// 	zoom - zoom level
		// 	max_iter - maximum number of iterations before a point is considered in the set
		mandelbrot(int H, int W, std::string center[], std::string zoom, int max_iter);
		
		// Changes the complex center of the image.
		// The new image is not rendered in this function 
		void changeCenter(std::string new_center[]);
		
		// Changes the zoom level of the image
		// The new image is not rendered in this function
		void changeZoom(std::string new_zoom);
		
		// Changes the maximum number of iterations
		// The new image is not rendered in this function
		void changeMaxIter(size_t new_max_iter);
		
		// Returns a cv array of the image with the parameters
		// previously specified by the user. If the image with
		// the current parameters hasn't been rendered, 
		// it is rendered in this step
		mandelbrot::ArrayCV getImageCV();
	
	private:
		int height;
		int width;
		
		std::string centerx, centery;
		std::string zoom;
		int max_iter;

    applyIterGPU GPU_object;
		
		bool isCalc=false;
		
		/*mandelbrot::Array cr;
		mandelbrot::Array ci;
		mandelbrot::Array zr;
		mandelbrot::Array zi;*/
		
		mandelbrot::Array values;
		mandelbrot::ArrayCV image;
		
		// Resets all variables of type mandelbrot::Array 
		// to zero except for cr and ci whose values are 
		// determined by the user defined parameters (i.e. center, zoom)
		void resetValues();
		
		// Renders the image
		void calcValues();
		
		// Avoids prominent color bands in rendered images
		//void smoothColor();
};

#endif
