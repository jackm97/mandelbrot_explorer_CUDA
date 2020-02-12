#include "applyIterGPU.h"

__global__
void GPU_PAR_FOR_HELPER(int height, int width,double* values, double* zr, double* zi, double* cr, double* ci, size_t max_iter)
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
      q = (cr[i*width+j]-1./4)*(cr[i*width+j]-1./4) + ci[i*width+j]*ci[i*width+j];

      if (q*(q+(cr[i*width+j]-1./4)) <= 1./4*ci[i*width+j]*ci[i*width+j])
        iters=max_iter;
      else if ((cr[i*width+j]+1)*(cr[i*width+j]+1) + ci[i*width+j]*ci[i*width+j] <= 1./16)
        iters=max_iter;
      while((zr2+zi2<=R2) && (iters<max_iter))
      {
        zi[i*width+j] = zi[i*width+j] * zr[i*width+j];
        zi[i*width+j] = zi[i*width+j] + zi[i*width+j] + ci[i*width+j];
        zr[i*width+j] = zr2 - zi2 + cr[i*width+j];
        zr2 = zr[i*width+j] * zr[i*width+j];
        zi2 = zi[i*width+j] * zi[i*width+j]; 
        iters++;
      }
      values[i*width+j] = iters;
    }
  }
}

void applyIterGPU::GPU_PAR_FOR(int height, int width)
{

  cudaMallocManaged(&values, height*width*sizeof(double));
  cudaMallocManaged(&zr, height*width*sizeof(double));
  cudaMallocManaged(&zi, height*width*sizeof(double));
  cudaMallocManaged(&cr, height*width*sizeof(double));
  cudaMallocManaged(&ci, height*width*sizeof(double));

  GPU_PAR_FOR_HELPER<<<1, 1024>>>(height, width, values, zr, zi, cr, ci, max_iter);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

