#include "mandelbrot.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char *argv[]){

	cout << argv[1] << endl;

	int height,width;
	mandelbrot::Point center;
	double x,y,zoom;
	size_t max_iter;

	cin >> height >> width;
	cin >> x >> y;
	cin >> zoom >> max_iter;

	center= mandelbrot::Point(x,y);
	height *= 2;
	width *= 2;
	mandelbrot a(height,width,center,zoom,max_iter);
	
	if (strcmp(argv[1],"0") == 0){
		mandelbrot::ArrayCV image = a.getImageCV();
		cv::applyColorMap(image, image, cv::COLORMAP_BONE);
        	cv::resize(image,image,cv::Size(),.5,.5);

		string fname = "hello";
                cv::namedWindow(fname, CV_WINDOW_NORMAL);
                cv::setWindowProperty(fname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                cv::imshow(fname,image);
                cv::waitKey(0);
                cv::destroyAllWindows();
	}
	
	else if(strcmp(argv[1],"1") == 0){
		for (int i=0; i<960*2; i++){
			string fname = "image" + to_string(i);
			zoom*=pow(10,.00625);
			a.changeZoom(zoom);
			mandelbrot::ArrayCV image = a.getImageCV();


			cv::applyColorMap(image, image, cv::COLORMAP_BONE);
        		cv::resize(image,image,cv::Size(),.5,.5);

                	string save_loc = "../images/";
			cv::imwrite(save_loc + fname + ".jpg",image);
		}
	}

	return 0;
}
	
