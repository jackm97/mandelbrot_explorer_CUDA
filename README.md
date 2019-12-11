# Mandelbrot Explorer

The Mandelbrot Set is one of the most well known fractals and I wanted to build a program that allows the user to explore this beautiful fractal. This repository contains the source code to build an application that can display single images or generate series of images to put together into animation sequences. The goal was to build an application that can render images quickly by using various optimizations such as parallel for loops and interior checking.

**Coming Soon: Paper with more detail on optimization techniques and their effectiveness**

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- OpenCV 4.1:
  - [Linux Install](https://docs.opencv.org/4.1.1/d7/d9f/tutorial_linux_install.html)
  - [Windows Install](https://www.learnopencv.com/install-opencv-4-on-windows/)
- Intel's Threading Building Blocks Library:
  - [Source Code](https://github.com/intel/tbb)
  - [Setup Environment Variables](https://software.intel.com/en-us/node/505529)
  - If tbb isn't detected, cmake will download source code and install it in the build directory
- GCC 8.3+
- cmake 3.13+

**NOTE: Lower versions of OpenCV will likely work, however it is recommended to use 4.1+ so that the user has access to all the most current colormaps. Lower versions of GCC and cmake will also likely work, however, it hasn't been tested. If you want to try a different version of cmake, make sure to edit the CMakeLists.txt to allow for earlier versions.**

### Installing
```
git clone https://github.com/jackm97/mandelbrot_explorer.git
cd ./mandelbrot_explorer
mkdir build
cd build
cmake .. && make
```
This should create an executable in the build directory called `mandelbrot_explorer`.

## Usage
To run the program call:
```
./mandelbrot_explorer arg1
```
`arg1` can take on one of two values: `0` to render a single image and `1` to render many images for an animation sequence.

### Example 1
<img src="./examples/example1.png" alt="drawing" width="640" height="360"/>

```
$ ./mandelbrot_explorer 0
Enter resolution, positive integers (e.g. height width): 1080 1920

Enter center point (e.g. x y): -1 0

Supersample? (Y/n): y

Enter zoom level(maximum of 1e12): 1.5

Enter iterations(positive integer): 5000
```
Notice that after the initial program execution, input prompts are displayed in the terminal. These particular inputs generate a 1080x1920 image centered on the complex coordinate (-1,0) with a zoom of 1.5x. The image is rendered at double the input resolution(e.g. supersampled) and then resized back to the input resolution using linear interpolation. This is an anti-aliasing method. The maximum number of iterations before a pixel is considered in the set is 5000.

### Example 2
<img src="./examples/example2.gif" alt="drawing" width="640" height="360"/>

```
$ ./mandelbrot_explorer 1
Enter resolution, positive integers (e.g. height width): 1080 1920

Enter center point (e.g. x y): -1.5231172841989 -4.5676519363104e-18

Supersample? (Y/n): n

Enter initial and final zoom level, maximum zoom is 1e12 (e.g. start_zoom end_zoom): 1 1e12

Enter number of frames to capture (positive integer): 1900

Enter iterations(positive integer): 5000

Enter file path to save image series: ../images2
```
Like the single image mode, the series image mode has input prompts. In this case the inputs shown above produce 1900 frames that range from a zoom of 1 to a zoom of 1e12. Each frame is 1080x1920 and is not supersampled. Once again, the maximum iterations are set to 5000. The file path for the generated frames is `../images2`.

To stitch the images into one mp4 file you could use a program like ffmpeg:
```
ffmpeg -r 60 -start_number 0 -i ../images2/image%d.jpg -crf 30 -vcodec libx264 -f mp4 /path/to/animation/animation.mp4
```
The above example generates a 60fps animation.

## Limitations/ Future Improvements
- Right now the zoom level is limited by the floating-point precision of the machine that the program is running on. I'd like to introduce the GMP library for arbitrary-precision
- Interior checking currently only identifies points within the cardioid or period-2 bulb. I'd like to implement more interior checking algorithms such as periodicity checking
- An [exponential map](https://mrob.com/pub/muency/exponentialmap.html) has the potential to speed up deep zooms by avoiding rendering points twice

## Authors

* **Jack Myers* - *Initial work* - [jackm97](https://github.com/jackm97)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
