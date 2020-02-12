#ifndef APPLYITERGPU_H
#define APPLYITERGPU_H

// Class for running Mandelbrot iteration in parallel
// within the mandelbrot class
class applyIterGPU {
    double *values, *zr, *zi, *cr, *ci;
    size_t max_iter;
public:  
  
  void GPU_PAR_FOR(int height, int width);
 
  applyIterGPU( double* values, double* zr, double* zi, double* cr, double* ci, size_t max_iter) :
      values(values),
    	zr(zr),
	    zi(zi),
	    cr(cr),
	    ci(ci),
	    max_iter(max_iter)
      {}
};

#endif
