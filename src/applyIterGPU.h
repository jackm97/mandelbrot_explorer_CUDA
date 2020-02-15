#ifndef APPLYITERGPU_H
#define APPLYITERGPU_H

// Class for running Mandelbrot iteration in parallel
// within the mandelbrot class
class applyIterGPU {
    double *values, *zr, *zi, *cr, *ci;
    size_t max_iter;
    int height, width;
public:  
  
  void GPU_PAR_FOR();

  void SET_COORD_VALS(double centerx, double centery, double zoom);

  void copyValues(double* target);
  
  void setMaxIter(size_t max_iter){this->max_iter=max_iter;}

  applyIterGPU(){}
 
  applyIterGPU( int height, int width, size_t max_iter);

  ~applyIterGPU();
};

#endif
