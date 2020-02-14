#include "applyIterGPU.h"

__global__
void GPU_PAR_FOR_HELPER(int height, int width,double* values, double* zr, double* zi, double* cr, double* ci, size_t max_iter)
{
  double iters=0,
         R2=1e6,
         zr2=0,
         zi2=0,
         q;
   
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (i<height*width)
  {
    // q is used to determine if a point is within the set
    // without needing to iterate to max_iter
    q = (cr[i]-1./4)*(cr[i]-1./4) + ci[i]*ci[i];

    if (q*(q+(cr[i]-1./4)) <= 1./4*ci[i]*ci[i])
      iters=max_iter;
    else if ((cr[i]+1)*(cr[i]+1) + ci[i]*ci[i] <= 1./16)
      iters=max_iter;
    while((zr2+zi2<=R2) && (iters<max_iter))
    {
      zi[i] = zi[i] * zr[i];
      zi[i] = zi[i] + zi[i] + ci[i];
      zr[i] = zr2 - zi2 + cr[i];
      zr2 = zr[i]* zr[i];
      zi2 = zi[i]* zi[i]; 
      iters++;
    }
    values[i] = iters;
  }
}

void applyIterGPU::GPU_PAR_FOR(int height, int width)
{
  double *GPUvalues, *GPUzr, *GPUzi, *GPUcr, *GPUci;
  
  cudaMalloc(&GPUvalues, height*width*sizeof(double));
  cudaMemcpy(GPUvalues, values, height*width*sizeof(double),cudaMemcpyHostToDevice);
  
  cudaMalloc(&GPUzr, height*width*sizeof(double));
  cudaMemcpy(GPUzr, zr, height*width*sizeof(double),cudaMemcpyHostToDevice);
  
  cudaMalloc(&GPUzi, height*width*sizeof(double));
  cudaMemcpy(GPUzi, zi, height*width*sizeof(double),cudaMemcpyHostToDevice);
  
  cudaMalloc(&GPUcr, height*width*sizeof(double));
  cudaMemcpy(GPUcr, cr, height*width*sizeof(double),cudaMemcpyHostToDevice);
  
  cudaMalloc(&GPUci, height*width*sizeof(double));
  cudaMemcpy(GPUci, ci, height*width*sizeof(double),cudaMemcpyHostToDevice);

  GPU_PAR_FOR_HELPER<<<((height*width)+255)/256, 256>>>(height, width, GPUvalues, GPUzr, GPUzi, GPUcr, GPUci, max_iter);

  cudaMemcpy(values, GPUvalues, height*width*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(zr, GPUzr, height*width*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(zi, GPUzi, height*width*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(cr, GPUcr, height*width*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(ci, GPUci, height*width*sizeof(double),cudaMemcpyDeviceToHost);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  cudaFree(GPUvalues);
  cudaFree(GPUzr);
  cudaFree(GPUzi);
  cudaFree(GPUcr);
  cudaFree(GPUci);
}

