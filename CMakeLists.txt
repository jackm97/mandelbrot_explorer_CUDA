cmake_minimum_required(VERSION 3.13)

project(mandelbrot_explorer LANGUAGES CXX CUDA)

#target_compile_features(mandelbrot_explorer PUBLIC cxx_std_11)

SET(GCC_COVERAGE_COMPILE_FLAGS "-O2 -std=c++11")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}${GCC_COVERAGE_LINK_FLAGS}")

find_package( OpenCV 4.1 REQUIRED )

# LOOK FOR TBB LIBRARY
# IF NOT FOUND, TBB IS DOWNLOADED AND BUILT FROM SOURCE
#find_library(TBB_LIB tbb)
#if (NOT TBB_LIB)
#	include(./cmake/TBBGet.cmake)
#	include(./cmake/TBBBuild.cmake)
#	tbb_get(TBB_ROOT tbb_root SOURCE_CODE)
#	tbb_build(TBB_ROOT ${tbb_root} CONFIG_DIR TBB_DIR)
#	find_package(TBB REQUIRED)
#	set(TBB_LIB ${TBB_IMPORTED_TARGETS})
#endif ()

#message(${TBB_LIB})
include_directories("./src/eigen3")

add_executable(mandelbrot_explorer "src/main.cpp" "src/mandelbrot.cpp" "src/applyIterGPU.cu" "src/applyIterGPU.h")
target_link_libraries( mandelbrot_explorer ${OpenCV_LIBS})# ${TBB_LIB})
