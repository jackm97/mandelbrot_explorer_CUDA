#include "applyIterGPU.h"

__device__
void calcPoint(float& cr, float& ci, float centerx, float centery, float zoom, int width, int height, int i, int j)
{
  float aspect_ratio = float(width)/height;
  float x_range, y_range, xmin, ymin, intervalx, intervaly;
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
  
  cr = xmin + i*intervalx;
  ci = ymin + j*intervaly;
}

__global__
void GPU_PAR_FOR_HELPER(int height, int width,float* values, float centerx, float centery, float zoom, size_t max_iter)
{
  float iters=0,
         R2=10,
         zr=0,
         zi=0,
         cr=0,
         ci=0,
         zr2=0,
         zi2=0;
   
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (idx<height*width)
  {
    float i = floorf(idx/height);
    float j = idx - height*i;
    calcPoint(cr,ci,centerx,centery,zoom,width,height,i,j);
    while((zr2+zi2<=R2) && (iters<max_iter))
    {
      zi = zi * zr;
      zi = zi + zi + ci;
      zr = zr2 - zi2 + cr;
      zr2 = zr* zr;
      zi2 = zi* zi; 
      iters++;
    }
    values[idx] = iters;
  }
}

applyIterGPU::applyIterGPU(int height, int width, size_t max_iter): 
  height(height),
  width(width),
  max_iter(max_iter)
{
  cudaMalloc(&values, height*width*sizeof(float));
  /*cudaMalloc(&zr, height*width*sizeof(float));
  cudaMalloc(&zi, height*width*sizeof(float));
  cudaMalloc(&cr, height*width*sizeof(float));
  cudaMalloc(&ci, height*width*sizeof(float));*/
}

applyIterGPU::~applyIterGPU()
{
  cudaFree(values);
  /*cudaFree(zr);
  cudaFree(zi);
  cudaFree(cr);
  cudaFree(ci);*/
}

void applyIterGPU::SET_COORD_VALS(float centerx, float centery, float zoom)
{
  this->centerx = centerx;
  this->centery = centery;
  this->zoom = zoom;
  //SET_COORD_VALS_HELPER<<<(width+127)/128, 128>>>(zr, zi, cr, ci, centerx, centery, zoom, width, height);
}
    
  
void applyIterGPU::GPU_PAR_FOR()
{  

  GPU_PAR_FOR_HELPER<<<((height*width)+255)/256, 256>>>(height, width, values, centerx, centery, zoom, max_iter);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void applyIterGPU::copyValues(float* target)
{
  cudaMemcpy(target, values, height*width*sizeof(float),cudaMemcpyDeviceToHost);
}

