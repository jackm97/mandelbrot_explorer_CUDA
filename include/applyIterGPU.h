#ifndef APPLYITERGPU_H
#define APPLYITERGPU_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>
#include <string>

// Class for running Mandelbrot iteration in parallel
// within the mandelbrot class
class applyIterGPU {
public:  
  
  void GPU_PAR_FOR();

  void moveTexture(int direction);

  void SET_COORD_VALS(std::string centerx, std::string centery, float zoom);

  void SET_ZOOM(float zoom);

  void getCenterString(std::string &centerx, std::string &centery);

  void copyValues(float* target);

  cudaGraphicsResource_t* getReferencePointer();

  void registerTextureResource(GLuint image);
  
  void setMaxIter(size_t max_iter){this->max_iter=max_iter;}

  applyIterGPU(){}
 
  applyIterGPU( int height, int width, size_t max_iter);

  ~applyIterGPU();

private:
    float center[2][5] = {{0.0,0.0,0.0,0.0,0.0},{0.0,0.0,0.0,0.0,0.0}};
    float zoom = 0;
    size_t max_iter;
    int height, width;
    float* iterData, *iterDataTmp;

    // OPENGL STUFF
    cudaGraphicsResource_t resource;
    cudaArray_t mappedArray;
};

#endif
