#include <cmath>
#include <iostream>
#include "applyIterGPU.h"
#include "multi_prec/multi_prec_certif.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <cuda_fp16.h>
#include <GLFW/glfw3.h>

surface<void, cudaSurfaceType2D> surfRef;

__device__
void smoothColor(float& iters, float zr2, float zi2)
{
  float nu;

  nu = zr2 + zi2;
  nu = log10f(nu);
  nu/=2;
  nu/=0.69314;
  nu = log10f(nu);
  nu/=0.69314;
  iters = iters + 1 - nu;
}

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
__global__ void
 __launch_bounds__(256, 4) 
GPU_PAR_FOR_HELPER(int height, int width, multi_prec<prec> centerx, multi_prec<prec> centery, multi_prec<prec> zoom, int max_iter)
{
  
  multi_prec<prec> cr=0., ci=0., q;

  multi_prec<prec> R2=10.,
                zr=0.,
                zi=0.,
                zr2=0.,
                zi2=0.;

  float iters = 0.;
   
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
      iters=NAN;
    else if ((cr+1)*(cr+1) + ci*ci <= 1./16)
      iters=NAN;
    
    while((zr2+zi2<=R2) && (iters<max_iter))
    {
      zi = zi * zr;
      zi = zi + zi + ci;
      zr = zr2 - zi2 + cr;
      zr2 = zr* zr;
      zi2 = zi* zi; 
      iters+=1;
    }

    if (iters == max_iter)
      iters = NAN;
    multi_prec<1> zr2_32(zr2.getData(),prec);
    multi_prec<1> zi2_32(zi2.getData(),prec);
    smoothColor(iters,zr2_32.getData()[0],zi2_32.getData()[0]);

    float value = iters/max_iter * float(iters != max_iter);
    value = fmodf(20 * value, 1.0) * 2;
    if ( value > 1)
      value = 2 - value;
    surf2Dwrite(value, surfRef, 4 * i, j);
    //printf("%.3f\n",cr);
  }
}

applyIterGPU::applyIterGPU(int height, int width, size_t max_iter): 
  height(height),
  width(width),
  max_iter(max_iter)
{
}

applyIterGPU::~applyIterGPU()
{
  /*cudaFree(zr);
  cudaFree(zi);
  cudaFree(cr);
  cudaFree(ci);*/
}

void applyIterGPU::SET_COORD_VALS(std::string centerx, std::string centery, float zoom)
{
  this->centerx = centerx;
  this->centery = centery;
  this->zoom = zoom;
  //SET_COORD_VALS_HELPER<<<(width+127)/128, 128>>>(zr, zi, cr, ci, centerx, centery, zoom, width, height);
}
    
template <int prec>
__host__  
void GPU_PAR_FOR_T(int height, int width, int max_iter, const char* centerx, const char* centery, float zoom_level)
{ 
  multi_prec<prec> centerx_ = centerx,
                centery_ = centery;

  // the zoom interval is designed to be exponential (i.e. a zoom interval of
  // one would have zoom levels of 1e0, 1e1, 1e2, etc.)
  float mod_exp = fmod(zoom_level,1.0);
  float base;
  if (mod_exp==0)
          base = 1;
  else
          base = pow(10,mod_exp);

  string zoom_str = to_string(int(floor(zoom_level)));
  zoom_str = "1e" + zoom_str;

  multi_prec<prec> zoom(zoom_str.c_str());
  zoom = (base*zoom);

  GPU_PAR_FOR_HELPER<prec><<<(height*width+255)/256, 256>>>(height, width, centerx_, centery_, zoom, max_iter);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void applyIterGPU::GPU_PAR_FOR()
{
  cudaGraphicsMapResources ( 1, &resource );
  cudaGraphicsSubResourceGetMappedArray ( &mappedArray, resource, 0, 0 );
  cudaBindSurfaceToArray(surfRef, mappedArray);
 float prec2 = 9.0, prec3 = 18.0, prec4 = 20.0;

  float zoom_check = zoom  + log10(float(max(height, width))) + log10(1000.);
  if (zoom_check>prec4){
    //std::cout << "\nUsing quadruple precision" << std::endl;
    GPU_PAR_FOR_T<4>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom);
  }
  else if (zoom_check>prec3){
    //std::cout << "Using triple precision" << std::endl;
    GPU_PAR_FOR_T<3>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom);
  }
  else if (zoom_check>prec2){
    //std::cout << "Using double precision" << std::endl;
    GPU_PAR_FOR_T<2>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom);
  }
  else{
    //std::cout << "Using single precision" << std::endl;
    GPU_PAR_FOR_T<1>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom);
  }
  cudaGraphicsUnmapResources( 1, &resource );
}

void applyIterGPU::copyValues(float* target)
{
  //cudaMemcpy(target, values, height*width*sizeof(float),cudaMemcpyDeviceToHost);
}

cudaGraphicsResource_t* applyIterGPU::getReferencePointer(){
	return &resource;
}

void applyIterGPU::registerTextureResource(GLuint image)
{
  cudaGraphicsGLRegisterImage( &resource, image, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
}