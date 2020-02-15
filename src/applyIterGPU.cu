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

    /*if (q*(q+(cr[i]-1./4)) <= 1./4*ci[i]*ci[i])
      iters=max_iter;
    else if ((cr[i]+1)*(cr[i]+1) + ci[i]*ci[i] <= 1./16)
      iters=max_iter;*/
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

__global__
void SET_COORD_VALS_HELPER(double* zr, double* zi, double* cr, double* ci, double centerx, double centery, double zoom, int width, int height)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  double aspect_ratio = double(width)/height;
  double x_range, y_range, xmin, ymin, intervalx, intervaly;
  if (aspect_ratio<1){
    x_range = 4/zoom;
    y_range = (1/aspect_ratio)*4/zoom;
  }
  else{
    x_range = (aspect_ratio)*4/zoom;
    y_range = 4/zoom;
  }
  xmin = centerx - x_range/2;
  ymin = centery - y_range/2;
  intervalx = x_range/width;
  intervaly = y_range/height;
  
  double x,y;
  if (i<width){
    for (int j=0; j<height; j++){
      x = xmin + i*intervalx;
      y = ymin + j*intervaly;
      //mandelbrot::Point point(xmin + i*intervalx, ymin + j*intervaly);
      cr[height*i + j] = x;
      ci[height*i + j] = y;
      zr[height*i + j] = 0;
      zr[height*i + j] = 0;
    }
  }
}

applyIterGPU::applyIterGPU(int height, int width, size_t max_iter): 
  height(height),
  width(width),
  max_iter(max_iter)
{
  cudaMalloc(&values, height*width*sizeof(double));
  cudaMalloc(&zr, height*width*sizeof(double));
  cudaMalloc(&zi, height*width*sizeof(double));
  cudaMalloc(&cr, height*width*sizeof(double));
  cudaMalloc(&ci, height*width*sizeof(double));
}

applyIterGPU::~applyIterGPU()
{
  cudaFree(values);
  cudaFree(zr);
  cudaFree(zi);
  cudaFree(cr);
  cudaFree(ci);
}

void applyIterGPU::SET_COORD_VALS(double centerx, double centery, double zoom)
{
  SET_COORD_VALS_HELPER<<<(width+255)/256, 256>>>(zr, zi, cr, ci, centerx, centery, zoom, width, height);
}
    
  
void applyIterGPU::GPU_PAR_FOR()
{  

  GPU_PAR_FOR_HELPER<<<((height*width)+255)/256, 256>>>(height, width, values, zr, zi, cr, ci, max_iter);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void applyIterGPU::copyValues(double* target)
{
  cudaMemcpy(target, values, height*width*sizeof(double),cudaMemcpyDeviceToHost);
}

