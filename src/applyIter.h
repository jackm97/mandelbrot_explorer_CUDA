#include "tbb/tbb.h"
#include "mandelbrot.h"

#ifndef APPLYITER_H
#define APPLYITER_H

// Class for running Mandelbrot iteration in parallel
// within the mandelbrot class
// 	Example:
// 	tbb::parallel_for(tbb::blocked_range2d<size_t>(0, height, 0, width), applyIter(values,zr,zi,cr,ci,max_iter));
//
class applyIter {
    mandelbrot::Array &values, &zr, &zi, &cr, &ci;
    size_t max_iter;
public:
    // Runs the iteration z_n = z_{n-1}^2 + c on each complex point c
    void operator()( const tbb::blocked_range2d<size_t>& r ) const {
        double iters=0,
               R2=1e6,
               zr2=0,
               zi2=0,
	       q;
        for( size_t i=r.cols().begin(); i!=r.cols().end(); ++i ){ 
           for (size_t j=r.rows().begin(); j!=r.rows().end(); ++j){
                        iters=0;
                        zr2=0;
                        zi2=0;

			// q is used to determine if a point is within the set
		   	// without needing to iterate to max_iter
			q = (cr(j,i)-1./4)*(cr(j,i)-1./4) + ci(j,i)*ci(j,i);
			
			if (q*(q+(cr(j,i)-1./4)) <= 1./4*ci(j,i)*ci(j,i))
				iters=max_iter;
			else if ((cr(j,i)+1)*(cr(j,i)+1) + ci(j,i)*ci(j,i) <= 1./16)
				iters=max_iter;
			
			while((zr2+zi2<=R2) && (iters<max_iter)){
                                zi(j,i) = zi(j,i) * zr(j,i);
                                zi(j,i) = zi(j,i) + zi(j,i) + ci(j,i);
                                zr(j,i) = zr2 - zi2 + cr(j,i);
                                zr2 = zr(j,i) * zr(j,i);
                                zi2 = zi(j,i) * zi(j,i);
                                iters++;
                        }
                        values(j,i) = iters;
	   }
	}
    }
  
__global__
void GPU_PAR_FOR(int height, int width)
{
  double iters=0,
         R2=1e6,
         zr2=0,
         zi2=0,
         q;
   
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int j=index; j<width; j+=stride)
  {
    for (int i=0; i<height; i++)
    {
      iters=0;
      zr2=0;
      zi2=0;
 
      // q is used to determine if a point is within the set
      // without needing to iterate to max_iter
      q = (cr(j,i)-1./4)*(cr(j,i)-1./4) + ci(j,i)*ci(j,i);

      if (q*(q+(cr(j,i)-1./4)) <= 1./4*ci(j,i)*ci(j,i))
        iters=max_iter;
      else if ((cr(j,i)+1)*(cr(j,i)+1) + ci(j,i)*ci(j,i) <= 1./16)
        iters=max_iter;
      while((zr2+zi2<=R2) && (iters<max_iter))
      {
        zi(j,i) = zi(j,i) * zr(j,i);
        zi(j,i) = zi(j,i) + zi(j,i) + ci(j,i);
        zr(j,i) = zr2 - zi2 + cr(j,i);
        zr2 = zr(j,i) * zr(j,i);
        zi2 = zi(j,i) * zi(j,i);
        iters++;
      }
      values(j,i) = iters;
    }
  }
}
           

    applyIter( mandelbrot::Array &values, mandelbrot::Array &zr, mandelbrot::Array &zi, mandelbrot::Array &cr, mandelbrot::Array &ci, size_t max_iter) :
        values(values),
    	zr(zr),
	zi(zi),
	cr(cr),
	ci(ci),
	max_iter(max_iter)
    {}
};

#endif
