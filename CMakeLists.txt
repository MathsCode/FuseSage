cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(fuseSage LANGUAGES CXX CUDA)
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
# find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(SRCS src/fuseSage_op.cpp src/fuseSage.cu src/kernel.cu src/utils.h src/agg.cu src/classify.cu src/sample.cu) 
## Define our library target
add_library(fuseSage SHARED ${SRCS})
## Enable C++11
# target_compile_features(op_demo PRIVATE cxx_range_for)
## Link against LibTorch
target_link_libraries(fuseSage "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" -lcublas -lcusparse)
# target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")



