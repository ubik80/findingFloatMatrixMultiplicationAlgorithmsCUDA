cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(PythonLibs 3.6 REQUIRED)

add_subdirectory(pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(backpropCUDA SHARED
  backpropCUDA.cpp
  backpropCUDA.cu)

target_link_libraries(
  backpropCUDA
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(backpropCUDA PROPERTIES PREFIX "")
