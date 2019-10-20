cmake_minimum_required(VERSION 3.13)

project(mandelbrot_explorer)

SET(GCC_COVERAGE_COMPILE_FLAGS "-fopenmp -march=native -msse2 -O2 -std=c++11")
SET(GCC_COVERAGE_LINK_FLAGS    "-ltbb")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}${GCC_COVERAGE_LINK_FLAGS}")

find_package( OpenCV REQUIRED )
include_directories("./src/eigen3")
add_executable(mandelbrot_explorer "src/main.cpp" "src/mandelbrot.cpp")
target_link_libraries( mandelbrot_explorer ${OpenCV_LIBS} ${CMAKE_EXE_LINKER_FLAGS})
