#define NDEBUG

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <opencv2/opencv.hpp>

#ifndef MANDELBROT_H
#define MANDELBROT_H
using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Index;
using namespace std;

class mandelbrot{
	public:
		mandelbrot(int H, int W, complex<double> center, double zoom, uint64_t max_iter);
		void calcValues();
		void createImage(string fname);
	private:
		uint64_t max_iter;
		int height;
		int width;
		Array<double,Dynamic,Dynamic> cr;
		Array<double,Dynamic, Dynamic> ci;
		Array<double,Dynamic,Dynamic> zr;
		Array<double,Dynamic,Dynamic> zi;
		Array<uint64_t,Dynamic,Dynamic> values;
};

#endif
