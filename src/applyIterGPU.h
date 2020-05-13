#ifndef APPLYITERGPU_H
#define APPLYITERGPU_H

#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <string>

// Class for running Mandelbrot iteration in parallel
// within the mandelbrot class
class applyIterGPU {
public:  
  
  void GPU_PAR_FOR();

  void SET_COORD_VALS(std::string centerx, std::string centery, std::string zoom);

  void copyValues(float* target);

  cudaGraphicsResource_t* getReferencePointer();

  void registerTextureResource(GLuint image);
  
  void setMaxIter(size_t max_iter){this->max_iter=max_iter;}

  applyIterGPU(){}
 
  applyIterGPU( int height, int width, size_t max_iter);

  ~applyIterGPU();

private:
    std::string centerx="0",centery="0",zoom="1";
    size_t max_iter;
    int height, width;

    // OPENGL STUFF
    cudaGraphicsResource_t resource;
    cudaArray_t mappedArray;
};

#endif
