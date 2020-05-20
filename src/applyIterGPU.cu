#include <cmath>
#include <iostream>
#include <string>
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
GPU_PAR_FOR_HELPER(int height, int width, multi_prec<prec> centerx, multi_prec<prec> centery, multi_prec<prec> zoom, int max_iter, float* iterData)
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
    iterData[idx] = value;
    surf2Dwrite(value, surfRef, 4 * i, j);
    //printf("%.3f\n",cr);
  }
}

applyIterGPU::applyIterGPU(int height, int width, size_t max_iter): 
  height(height),
  width(width),
  max_iter(max_iter)
{
  cudaMalloc(&iterData, height * width * sizeof(float));
  cudaMalloc(&iterDataTmp, height * width * sizeof(float));
}

applyIterGPU::~applyIterGPU()
{
  /*cudaFree(zr);
  cudaFree(zi);
  cudaFree(cr);
  cudaFree(ci);*/
  cudaFree(iterData);
  cudaFree(iterDataTmp);
}

void applyIterGPU::SET_COORD_VALS(std::string centerx, std::string centery, float zoom)
{
  std::string float_str = "";
  int current_prec = 0;
  int last = 0;
  while (current_prec < 5 && last < centerx.length()){
    while (last < centerx.length()){
      if (centerx[last]==' ') {last+=1; break;}
      else {float_str+=centerx[last]; last+=1;}
    }
    center[0][current_prec] = std::atof(float_str.c_str());
    float_str = "";
    current_prec += 1;
  }

  current_prec = 0;
  last = 0;
  while (current_prec < 5 && last < centerx.length()){
    while (last < centery.length()){
      if (centery[last]==' ') {last+=1; break;}
      else {float_str+=centery[last]; last+=1;}
    }
    center[1][current_prec] = std::atof(float_str.c_str());
    float_str = "";
    current_prec += 1;
  }

  this->zoom = zoom;
  //SET_COORD_VALS_HELPER<<<(width+127)/128, 128>>>(zr, zi, cr, ci, centerx, centery, zoom, width, height);
}

void applyIterGPU::SET_ZOOM(float zoom)
{
  this->zoom = zoom;
}

void applyIterGPU::getCenterString(std::string &centerx, std::string &centery)
{
  multi_prec<5> cr , ci;
  cr.setData(center[0],5);
  ci.setData(center[1],5);

  centerx = cr.prettyPrintBF();
  centery = ci.prettyPrintBF();
}
    
template <int prec>
__host__  
void GPU_PAR_FOR_T(int height, int width, int max_iter, float center[2][5], float zoom_level, float* iterData)
{ 
  multi_prec<prec> centerx_ , centery_;
  centerx_.setData(center[0],prec);
  centery_.setData(center[1],prec);

  // the zoom interval is designed to be exponential (i.e. a zoom interval of
  // one would have zoom levels of 1e0, 1e1, 1e2, etc.)
  float mod_exp = fmod(zoom_level,1.0);
  float base;
  if (mod_exp==0)
          base = 1;
  else
          base = pow(10,mod_exp);

  std::string zoom_str = to_string(int(floor(zoom_level)));
  zoom_str = "1e" + zoom_str;

  multi_prec<prec> zoom(zoom_str.c_str());
  zoom = (base*zoom);

  GPU_PAR_FOR_HELPER<prec><<<(height*width+255)/256, 256>>>(height, width, centerx_, centery_, zoom, max_iter, iterData);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

//---------------------------------------------------------
//Class Methods

template<int prec>
__global__ void
__launch_bounds__(256, 4) 
rotateImage(int direction, int height, int width, float* iterData, float* iterDataTmp)
{
     
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (idx < height*width)
  {
    float i = idx/height;
    float j = idx - height*i;
    
    // up
    if (direction == 1)
    {
      if (j + 1 == height){
        j = 0;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
      else{
        j += 1;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
    }        
    // down
    else if (direction == 0)
    {
      if (j - 1 < 0){
        j = height - 1;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
      else{
        j-=1;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
    }        
    // left
    else if (direction == 3)
    {
      if (i - 1 < 0){
        i = width - 1;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
      else{
        i -= 1;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
    }        
    // right
    else if (direction == 2)
    {
      if (i + 1 == width){
        i = 0;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
      else{
        i += 1;
        iterDataTmp[int(height*i + j)] = iterData[idx];
        surf2Dwrite(iterData[idx],surfRef,4*i,j);
      }
    }   
  }
}

template <int prec>
__host__
void moveCenter(int direction, int height, int width, int max_iter, float new_center[2][5], multi_prec<prec> zoom)
{
  multi_prec<prec> cr , ci;
  cr.setData(new_center[0],prec);
  ci.setData(new_center[1],prec);

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
  
  // up
  if (direction == 0)
    ci += y_range/height;
  // down
  else if (direction == 1)
    ci -= y_range/height;
  // left
  else if (direction == 2)
    cr -= x_range/width;
  // down
  else if (direction == 3)
    cr += x_range/width;

  // std::string cr_s = cr.prettyPrintBF();
  // std::string ci_s = ci.prettyPrintBF();

  // std::string c_s_array[2] = {cr_s,ci_s};

  // new_center[0] = "";
  // new_center[1] = "";
  // for (int k=0; k<2; k++){
  //   int last = 0;
  //   std::string digits, sign;
  //   int exp;
  //   std::string c_s = c_s_array[k];
  //   for (int i=0; i<prec; i++){
  //     if (c_s[last] == '-'){
  //       sign = "-";
  //       last += 1;
  //     }
  //     else
  //       sign = "+";
      
  //     digits = "";
  //     while (c_s[last] != 'e'){
  //       if (c_s[last]!='.')
  //         digits += c_s[last];
  //       last+=1;
  //     }
  //     last +=1;
      
  //     std::string exp_s = "";
  //     while (c_s[last]!=' '){
  //       exp_s += c_s[last];
  //       last+=1;
  //     }
  //     last+=1;
  //     exp = std::stoi(exp_s);

  //     new_center[k] += sign;
  //     if (exp < 0){
  //       new_center[k] += '.';
  //       for (int j=0; j<-1*exp - 1; j++)
  //         new_center[k]+='0';
  //       for (int j=0; j<digits.length(); j++)
  //         new_center[k]+=digits[j];
  //     }
  //     else if (exp >= 0){
  //       for (int j=0; j<=exp+1 || j<digits.length(); j++){
  //         if (j>digits.length()){
  //           if (j==exp+1){new_center[k]+='.';j-=1;exp=-1;}
  //           else new_center[k]+='0';
  //         }
  //         else if (j==exp+1){new_center[k]+='.';j-=1;exp=-1;}
  //         else new_center[k]+=digits[j];
  //       }
  //     }
  //   }
  //   while (last<c_s.length())last++;
  // }

  for (int i=0; i<prec; i++){
    new_center[0][i] = cr.getData()[i];
    new_center[1][i] = ci.getData()[i];
  }
}

template<int prec>
__global__ void
__launch_bounds__(256, 4) 
renderRow(float row, int height, int width, int max_iter, multi_prec<prec> centerx, multi_prec<prec> centery, multi_prec<prec> zoom, float* iterData)
{
  
  multi_prec<prec> cr=0., ci=0., q;

  multi_prec<prec> R2=10.,
                zr=0.,
                zi=0.,
                zr2=0.,
                zi2=0.;

  float iters = 0.;
   
  float col = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (col < width)
  {
    iters=0.;
    zr=0.;
    zi=0.;
    cr=0.;
    ci=0.;
    zr2=0.;
    zi2=0.;
    
    calcPoint(cr,ci,centerx,centery,zoom,width,height,col,row);
    
    
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
    iterData[int(height * col + row)] = value;
    surf2Dwrite(value, surfRef, 4 * col, row);
    //printf("%.3f\n",cr);
  }
}

template<int prec>
__global__ void
__launch_bounds__(256, 4) 
renderCol(float col, int height, int width, int max_iter, multi_prec<prec> centerx, multi_prec<prec> centery, multi_prec<prec> zoom, float* iterData)
{
  
  multi_prec<prec> cr=0., ci=0., q;

  multi_prec<prec> R2=10.,
                zr=0.,
                zi=0.,
                zr2=0.,
                zi2=0.;

  float iters = 0.;
   
  float row = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (row < height)
  {
    iters=0.;
    zr=0.;
    zi=0.;
    cr=0.;
    ci=0.;
    zr2=0.;
    zi2=0.;
    
    calcPoint(cr,ci,centerx,centery,zoom,width,height,col,row);
    
    
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
    iterData[int(height * col + row)] = value;
    surf2Dwrite(value, surfRef, 4 * col, row);
    //printf("%.3f\n",cr);
  }
}

template <int prec>
__host__
void moveTexture_T(int direction, int height, int width, int max_iter, float center[2][5], float zoom_level, float* iterData, float* iterDataTmp)
{ 

  // the zoom interval is designed to be exponential (i.e. a zoom interval of
  // one would have zoom levels of 1e0, 1e1, 1e2, etc.)
  float mod_exp = fmod(zoom_level,1.0);
  float base;
  if (mod_exp==0)
          base = 1;
  else
          base = pow(10,mod_exp);

  std::string zoom_str = to_string(int(floor(zoom_level)));
  zoom_str = "1e" + zoom_str;

  multi_prec<prec> zoom(zoom_str.c_str());
  zoom = (base*zoom);

  // rotate image
  rotateImage<prec><<<(height*width+255)/256, 256>>>(direction, height, width, iterData, iterDataTmp);
  cudaDeviceSynchronize();
  cudaMemcpy(iterData,iterDataTmp,height*width*sizeof(float),cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize(); 

  // move center
  moveCenter<prec>(direction, height, width, max_iter, center, zoom);
  multi_prec<prec> centerx_ , centery_;
  centerx_.setData(center[0],prec);
  centery_.setData(center[1],prec);

  // up
  if (direction == 1)
    renderRow<prec><<<(width+255)/256, 256>>>(0, height, width, max_iter, centerx_, centery_, zoom, iterData);
  // down
  else if (direction == 0)
    renderRow<prec><<<(width+255)/256, 256>>>(height - 1, height, width, max_iter, centerx_, centery_, zoom, iterData);
  // left
  else if (direction == 3)
    renderCol<prec><<<(height+255)/256, 256>>>(width - 1, height, width, max_iter, centerx_, centery_, zoom, iterData);
  // up
  else if (direction == 2)
    renderCol<prec><<<(height+255)/256, 256>>>(0, height, width, max_iter, centerx_, centery_, zoom, iterData);
    
  cudaDeviceSynchronize();

}

void applyIterGPU::moveTexture(int direction)
{
  cudaGraphicsMapResources ( 1, &resource );
  cudaGraphicsSubResourceGetMappedArray ( &mappedArray, resource, 0, 0 );
  cudaBindSurfaceToArray(surfRef, mappedArray);
  float prec2 = 9.0, prec3 = 18.0, prec4 = 20.0;

  float zoom_check = zoom  + log10(float(max(height, width))) + log10(1000.);
  if (zoom_check>prec4){
    moveTexture_T<4>(direction, height, width, max_iter, center, zoom, iterData, iterDataTmp);
  }
  else if (zoom_check>prec3){
    moveTexture_T<3>(direction, height, width, max_iter, center, zoom, iterData, iterDataTmp);
  }
  else if (zoom_check>prec2){
    moveTexture_T<2>(direction, height, width, max_iter, center, zoom, iterData, iterDataTmp);
  }
  else {
    moveTexture_T<1>(direction, height, width, max_iter, center, zoom, iterData, iterDataTmp);
  }

  cudaGraphicsUnmapResources( 1, &resource );
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
    GPU_PAR_FOR_T<4>(height,width,max_iter, center,zoom, iterData);
  }
  else if (zoom_check>prec3){
    //std::cout << "Using triple precision" << std::endl;
    GPU_PAR_FOR_T<3>(height,width,max_iter, center,zoom, iterData);
  }
  else if (zoom_check>prec2){
    //std::cout << "Using double precision" << std::endl;
    GPU_PAR_FOR_T<2>(height,width,max_iter, center,zoom, iterData);
  }
  else{
    //std::cout << "Using single precision" << std::endl;
    GPU_PAR_FOR_T<1>(height,width,max_iter, center,zoom, iterData);
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