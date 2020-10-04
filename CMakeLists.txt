cmake_minimum_required(VERSION 3.13)

project(mandelbrot_explorer LANGUAGES CXX CUDA)

include_directories(include "./extern")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("./extern/glfw")

add_library(lodepng STATIC "./extern/lodepng-master/lodepng.cpp")

add_library(glad STATIC "src/glad.cpp")

add_library(mandelbrot STATIC "src/mandelbrot.cpp" "src/applyIterGPU.cu")
target_link_libraries(mandelbrot PUBLIC glad)

add_library(Shader STATIC "src/Shader.cpp")

add_executable(mandelbrot_explorer "src/main.cpp") 
set_property(TARGET mandelbrot_explorer PROPERTY CUDA_STANDARD 11)
target_link_libraries( mandelbrot_explorer PUBLIC glad	
					     PUBLIC mandelbrot
					     PUBLIC glfw
					     PUBLIC Shader
						 PUBLIC lodepng
					     )

