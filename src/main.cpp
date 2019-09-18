#include "mandelbrot.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>

using namespace std;

template<class var_type>
void input_check(string prompt_str, int argc, var_type argv[]);

int main(int argc, char *argv[]){

	string USAGE = \
		       "USAGE: ./mandelbrot_explorer arg1\n   arg1:  \"0\" for single image, \"1\" for zoom animation\n";
			
	if (argc!=2){
		cerr << USAGE;
		exit(1);
	}
	if (strcmp(argv[1],"0")!=0 && strcmp(argv[1],"1")!=0){
	       cerr << USAGE;	
	       exit(1);
	}

	int resolution[2];
	double center[2];
	char supersample[1];

	cout << "Enter resolution (e.g. height width): ";
	cin >> resolution[0] >> resolution[1];
	input_check("Enter resolution, positive integers (e.g. height width): ", 2, resolution);
	while (resolution[0] <= 0 || resolution[1] <= 0){
		cout << "You have entered the wrong input" << endl;
		cout << "Enter resolution, positive integers (e.g. height width): ";
		cin >> resolution[0] >> resolution[1];
	        input_check("Enter resolution, positive integers (e.g. height width): ", 2, resolution);
	}

	
	cout << endl << "Enter center point (e.g. x y): ";
	cin >> center[0] >> center[1];
	input_check("Enter center point (e.g. x y): ", 2, center);

	cout << endl << "Supersample? (Y/n): ";
	cin >> supersample[0];
	input_check("Supersample? (Y/n): ", 1, supersample);
	while (supersample[0]!='y' && supersample[0]!='Y' && supersample[0]!='n' && supersample[0]!='N'){
		cout << "You have entered the wrong input" << endl;
                cout << "Supersample? (Y/n): ";
		cin >> supersample;
		input_check("Supersample? (Y/n): ", 1, supersample);
	}
	if (supersample[0]=='y' || supersample[0]=='Y'){
		resolution[0]*=2;
		resolution[1]*=2;
	}


		
	if (strcmp(argv[1],"0") == 0){
		double zoom[1];
		int64_t max_iter[1];
		
		cout << endl <<  "Enter zoom level(maximum of 1e12): ";
		cin >> zoom[0];
		input_check("Enter zoom level(maximum of 1e12): ", 1, zoom);
		while (zoom[0] > 1e12 || zoom[0] <= 0){
			cout << "You have entered the wrong input" << endl;
			cout << "Enter zoom level(maximum of 1e12): ";
			cin >> zoom[0];
			input_check("Enter zoom level(maximum of 1e12): ", 1, zoom);
		}


		cout << endl << "Enter iterations(positive integer): ";
		cin >> max_iter[0];
		input_check("Enter iterations(positive integer): ", 1, max_iter);
		while (max_iter[0] <= 0){
                        cout << "You have entered the wrong input" << endl;
                        cout << "Enter iterations(positive integer): ";
                        cin >> max_iter[0];
                        input_check("Enter iterations(positive integer): ", 1, max_iter);
                }


		mandelbrot m(resolution[0], resolution[1], mandelbrot::Point(center[0],center[1]), zoom[0], max_iter[0]);
		mandelbrot::ArrayCV image = m.getImageCV();
		cv::applyColorMap(image, image, cv::COLORMAP_BONE);
		if (supersample[0]=='y' || supersample[0]=='Y')
        		cv::resize(image,image,cv::Size(),.5,.5);

		string fname = "";
                cv::namedWindow(fname, CV_WINDOW_NORMAL);
                //cv::setWindowProperty(fname, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
                cv::imshow(fname,image);
                cv::waitKey(0);
                cv::destroyAllWindows();
	}
	
	else if(strcmp(argv[1],"1") == 0){
		double zoom, zoom_range[2], frame_count[1];
		int64_t max_iter[1];

		cout << endl << "Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ";
		cin >> zoom_range[0] >> zoom_range[1];
		input_check("Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ", 2, zoom_range);
		while (zoom_range[0] > 1e12 || zoom_range[0] <= 0 || zoom_range[1] > 1e12 || zoom_range[1] <= 0){
			cout << "You have entered the wrong input" << endl;
			cout << "Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ";
			cin >> zoom_range[0] >> zoom_range[1];
			input_check("Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): ", 2, zoom_range);
		}

		cout << endl << "Enter number of frames to capture (positive integer): ";
		cin >> frame_count[0];
		input_check("Enter number of frames to capture (positive integer): ", 1, frame_count);
		while (frame_count[0] <= 0 || fmod(frame_count[0],int64_t(frame_count[0]))!=0){
			cout << "You have entered the wrong input" << endl;
			cout << "Enter number of frames to capture (positive integer): ";
			cin >> frame_count[0];
			input_check("Enter number of frames to capture (positive integer): ", 1, frame_count);
		}

		cout << endl << "Enter iterations(positive integer): ";
		cin >> max_iter[0];
                input_check("Enter iterations(positive integer): ", 1, max_iter);
		while (max_iter[0] <= 0){
			cout << "You have entered the wrong input" << endl;
			cout << "Enter iterations(positive integer): ";
			cin >> max_iter[0];
			input_check("Enter iterations(positive integer): ", 1, max_iter);
		}

		double zoom_interval;
		zoom_interval = (log10(zoom_range[1]) - log10(zoom_range[0]))/(frame_count[0]);
		
		zoom=zoom_range[0];
		mandelbrot m(resolution[0], resolution[1], mandelbrot::Point(center[0],center[1]), zoom, max_iter[0]);
		for (int i=0; i<frame_count[0]; i++){
			string fname = "image" + to_string(i);
			m.changeZoom(zoom);
			mandelbrot::ArrayCV image = m.getImageCV();

			cv::applyColorMap(image, image, cv::COLORMAP_BONE);
			if (supersample[0]=='y' || supersample[0]=='Y')
        			cv::resize(image,image,cv::Size(),.5,.5);

                	string save_loc = "../images/";
			cv::imwrite(save_loc + fname + ".jpg",image);
			zoom*=pow(10,zoom_interval);
		}
	}

	return 0;
}

template <class var_type>
void input_check(string prompt_str, int argc, var_type argv[]){
	while (1){
		if (cin.fail()){
			cin.clear();
			cin.ignore(numeric_limits<streamsize>::max(),'\n');
			cout << "You have entered the wrong input" << endl;
			cout << prompt_str;
			for (int i=0; i<argc; i++)
				cin >> argv[i];
		}
		else
			return;
	}
}
