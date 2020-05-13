cmake_minimum_required(VERSION 3.13)

project(mandelbrot_explorer LANGUAGES CXX CUDA)

#target_compile_features(mandelbrot_explorer PUBLIC cxx_std_11)

SET(GCC_COVERAGE_COMPILE_FLAGS "-O2 -std=c++11")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}${GCC_COVERAGE_LINK_FLAGS}")

find_package( OpenCV 4.1 REQUIRED )

include_directories("./src/include/eigen3" "./src/include/")

add_executable(mandelbrot_explorer "src/main.cpp" "src/glad.cpp" "src/mandelbrot.cpp" "src/Shader.cpp" "src/applyIterGPU.cu" "src/applyIterGPU.h") 
target_link_libraries( mandelbrot_explorer ${OpenCV_LIBS} "-lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl")

