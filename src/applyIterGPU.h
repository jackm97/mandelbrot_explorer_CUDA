#ifndef APPLYITERGPU_H
#define APPLYITERGPU_H

// Class for running Mandelbrot iteration in parallel
// within the mandelbrot class
class applyIterGPU {
    float *values, *zr, *zi, *cr, *ci;
    float centerx=0,centery=0,zoom=1;
    size_t max_iter;
    int height, width;
public:  
  
  void GPU_PAR_FOR();

  void SET_COORD_VALS(float centerx, float centery, float zoom);

  void copyValues(float* target);
  
  void setMaxIter(size_t max_iter){this->max_iter=max_iter;}

  applyIterGPU(){}
 
  applyIterGPU( int height, int width, size_t max_iter);

  ~applyIterGPU();
};

#endif
