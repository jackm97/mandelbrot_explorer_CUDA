#include <iostream>
#include "applyIterGPU.h"
#include "multi_prec/multi_prec_certif.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
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
GPU_PAR_FOR_HELPER(int height, int width, multi_prec<prec> centerx, multi_prec<prec> centery, multi_prec<prec> zoom, size_t max_iter)
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
      iters+=1;
    }
    multi_prec<1> zr2_32(zr2.getData(),prec);
    multi_prec<1> zi2_32(zi2.getData(),prec);
    smoothColor(iters,zr2_32.getData()[0],zi2_32.getData()[0]);
    //values[idx] = iters;
    float4 pixelVal;
    pixelVal.x = float(iters)/float(max_iter) * float(iters != max_iter);
    pixelVal.x = fmodf(20 * pixelVal.x, 1.0) * 2;
    if ( pixelVal.x > 1)
      pixelVal.x = 2 - pixelVal.x;
    pixelVal.y = 0.0f;
    pixelVal.z = 0.0f;
    pixelVal.w = 0.0f;
    surf2Dwrite(pixelVal, surfRef, 4 * 4 * i, j);
    //printf("%.3f\n",iters);
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

void applyIterGPU::SET_COORD_VALS(std::string centerx, std::string centery, std::string zoom)
{
  this->centerx = centerx;
  this->centery = centery;
  this->zoom = zoom;
  //SET_COORD_VALS_HELPER<<<(width+127)/128, 128>>>(zr, zi, cr, ci, centerx, centery, zoom, width, height);
}
    
template <int prec>  
void GPU_PAR_FOR_T(int height, int width, int max_iter, const char* centerx, const char* centery, const char* zoom)
{ 
  multi_prec<prec> centerx_ = centerx,
                centery_ = centery,
                zoom_ = zoom; 

  GPU_PAR_FOR_HELPER<prec><<<(height*width+255)/256, 256>>>(height, width, centerx_, centery_, zoom_, max_iter);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

void applyIterGPU::GPU_PAR_FOR()
{
  cudaGraphicsMapResources ( 1, &resource );
  cudaGraphicsSubResourceGetMappedArray ( &mappedArray, resource, 0, 0 );
  cudaBindSurfaceToArray(surfRef, mappedArray);
  multi_prec<5> zoom_ = zoom.c_str(),
                prec2("1e9"),
                prec3("1e18"),
                prec4("1e20");
  zoom_ = zoom_  * max(height, width);
  if (zoom_>prec4){
    //std::cout << "\nUsing quadruple precision" << std::endl;
    GPU_PAR_FOR_T<4>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
  }
  else if (zoom_>prec3){
    //std::cout << "Using triple precision" << std::endl;
    GPU_PAR_FOR_T<3>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
  }
  else if (zoom_>prec2){
    //std::cout << "Using double precision" << std::endl;
    GPU_PAR_FOR_T<2>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
  }
  else{
    //std::cout << "Using single precision" << std::endl;
    GPU_PAR_FOR_T<1>(height,width,max_iter,centerx.c_str(),centery.c_str(),zoom.c_str());
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