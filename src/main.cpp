#include "mandelbrot.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;

int main(int argc, char *argv[]){

	int height,width;
	mandelbrot::Point center;
	double x,y;
	char supersample;

	cout << "Enter resolution (e.g. height width): ";
	cin >> height >> width;
	
	cout << endl << "Enter center point (e.g. x y): ";
	cin >> x >> y;	
	center= mandelbrot::Point(x,y);

	cout << endl << "Supersample? (Y/n): ";
	cin >> supersample;
	if (supersample=='y' || supersample=='Y'){
		height*=2;
		width*=2;
	}

		
	if (strcmp(argv[1],"0") == 0){
		double zoom;
		size_t max_iter;
		
		cout << endl <<  "Enter zoom level(maximum of 1e12): ";
		cin >> zoom;

		cout << endl << "Enter iterations(integer): ";
		cin >> max_iter;

		mandelbrot m(height, width, mandelbrot::Point(x,y), zoom, max_iter);
		mandelbrot::ArrayCV image = m.getImageCV();
		cv::applyColorMap(image, image, cv::COLORMAP_BONE);
		//cv::GaussianBlur(image, image, cv::Size(3,3), 0, 0);
		if (supersample=='y' || supersample=='Y')
        		cv::resize(image,image,cv::Size(),.5,.5);

		string fname = "";
                cv::namedWindow(fname, CV_WINDOW_NORMAL);
                cv::setWindowProperty(fname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                cv::imshow(fname,image);
                cv::waitKey(0);
                cv::destroyAllWindows();
	}
	
	else if(strcmp(argv[1],"1") == 0){
		double zoom, zoom_start, zoom_end, anim_length;
		size_t max_iter;

		cout << endl << "Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ";
		cin >> zoom_start >> zoom_end;

		cout << endl << "Enter animation length in seconds: ";
		cin >> anim_length;

		cout << endl << "Enter iterations(positive integer): ";
		cin >> max_iter;

		double zoom_interval;
		zoom_interval = (log10(zoom_end) - log10(zoom_start))/(32*anim_length);
		
		zoom=zoom_start;
		mandelbrot m(height, width, mandelbrot::Point(x,y), zoom_start, max_iter);
		for (int i=0; i<int(32*anim_length); i++){
			string fname = "image" + to_string(i);
			m.changeZoom(zoom);
			mandelbrot::ArrayCV image = m.getImageCV();

			cv::applyColorMap(image, image, cv::COLORMAP_BONE);
			//cv::GaussianBlur(image, image, cv::Size(3,3), 0, 0);
			if (supersample=='y' || supersample=='Y')
        			cv::resize(image,image,cv::Size(),.5,.5);

                	string save_loc = "../images/";
			cv::imwrite(save_loc + fname + ".jpg",image);
			zoom*=pow(10,zoom_interval);
		}
	}

	return 0;
}
	
