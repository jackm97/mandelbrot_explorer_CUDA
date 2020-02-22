#include "applyIterGPU.h"
#include "multi_prec/multi_prec_certif.h"

template<int prec>
__device__
void calcPoint(multi_prec<prec>& cr, multi_prec<prec>& ci, multi_prec<prec>& centerx, multi_prec<prec>& centery, multi_prec<prec>& zoom, int& width, int& height, float& i, float& j)
{
  float aspect_ratio = float(width)/height;
  multi_prec<prec> x_range, y_range;
  if (aspect_ratio<1){
    x_range = 4/zoom;
    y_range = (1/aspect_ratio)*4/zoom;
  }
  else{
    x_range = (aspect_ratio)*4/zoom;
    y_range = 4/zoom;
  }
  
  cr = (centerx - x_range/2) + i*(x_range/width);
  ci = (centery - y_range/2) + j*(y_range/height);
}

template<int prec>
__global__
void GPU_PAR_FOR_HELPER(int height, int width,float* values, multi_prec<prec> centerx, multi_prec<prec> centery, multi_prec<prec> zoom, size_t max_iter)
{
  multi_prec<prec> R2=10.,
         zr=0.,
         zi=0.,
         cr=0.,
         ci=0.,
         zr2=0.,
         zi2=0.,
         q;

  float iters = 0;
   
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (idx < height*width)
  {
    iters=0.;
    zr=0.;
    zi=0.;
    cr=0.;
    ci=0.;
    zr2=0.;
    zi2=0.;
    
    float i = idx/height;
    float j = idx - height*i;
    calcPoint(cr,ci,centerx,centery,zoom,width,height,i,j);
    
    // q is used to determine if a point is within the set
    // without needing to iterate to max_iter
    q = (cr-1./4)*(cr-1./4) + ci*ci;

    if (q*(q+(cr-1./4)) <= 1./4*ci*ci)
      iters=max_iter;
    else if ((cr+1)*(cr+1) + ci*ci <= 1./16)
      iters=max_iter;
    
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

void applyIterGPU::SET_COORD_VALS(std::string centerx, std::string centery, std::string zoom)
{
  this->centerx = centerx;
  this->centery = centery;
  this->zoom = zoom;
  //SET_COORD_VALS_HELPER<<<(width+127)/128, 128>>>(zr, zi, cr, ci, centerx, centery, zoom, width, height);
}
    
template <int prec>  
void GPU_PAR_FOR_T(float* values, int height, int width, int max_iter, const char* centerx, const char* centery, const char* zoom)
{ 
  multi_prec<prec> centerx_ = centerx,
                centery_ = centery,
                zoom_ = zoom; 

  GPU_PAR_FOR_HELPER<prec><<<(height*width+255)/256, 256>>>(height, width, values, centerx_, centery_, zoom_, max_iter);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void applyIterGPU::GPU_PAR_FOR()
{
  multi_prec<5> zoom_ = zoom.c_str(),
                prec2 = "1e4",
                prec3 = "1e15",
                prec4 = "1e23";
  if (zoom_>prec2)
    GPU_PAR_FOR_T<2>(values,height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
  else if (zoom_>prec3)
    GPU_PAR_FOR_T<3>(values,height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
  else if (zoom_>prec4)
    GPU_PAR_FOR_T<4>(values,height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
  else
    GPU_PAR_FOR_T<1>(values,height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
}

void applyIterGPU::copyValues(float* target)
{
  cudaMemcpy(target, values, height*width*sizeof(float),cudaMemcpyDeviceToHost);
}

