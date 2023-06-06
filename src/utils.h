#include <cuda_runtime.h>
#include <torch/all.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
template <typename T>
inline T *get_ptr(torch::Tensor &t) 
{
    return reinterpret_cast<T *>(t.data_ptr());
}
template <typename T>
inline const T *get_ptr(const torch::Tensor &t) 
{
    return reinterpret_cast<T *>(t.data_ptr());
}       

template <class T>
void print_check(T *device_address,int number)
{
    T *host_data_I = nullptr;
    host_data_I = (T *)malloc(number*sizeof(T));
    cudaMemcpy(host_data_I,device_address,number*sizeof(T),cudaMemcpyDeviceToHost);
    for(int i = 0; i < number; i++)
    {
        std::cout<<host_data_I[i]<<' ';
    }
    std::cout<<'\n';
}
