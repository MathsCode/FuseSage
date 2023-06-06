#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


__global__ void addbiasrelu0(float *dst, int GPU_nodes,
                         float *GPU_update, float *GPU_upward,
                         float *CPU_update, float *CPU_upward,float *bias,
                         int *GPU_map,int *CPU_map,
                         int numElements,int feature_size);
__global__ void addbiasrelu1(float *dst, int sample_size, 
                         float *GPU_update, float *GPU_upward,
                         float *CPU_update, float *CPU_upward, float *bias,
                         int *GPU_csr_indptr,int*CPU_csr_indptr,
                         int numElements,int feature_size);
__global__ void addbiasrelu2(float *dst,int nElems,float *update,float *upward,float *bias,int feature_size);
__global__ void GPU_feature_transfer(float *dst,float *src,int32_t *id_map,int32_t * node_id,int num_nodes,int feature_size);
// __global__ void print_kernel(float *data,int numbers);
// __global__ void print_kernel(int32_t *data,int numbers);