#include <cuda_runtime.h>
#include "../include/kernel.h"
__global__ void addbiasrelu0(float *dst, int GPU_nodes,
                         float *GPU_update, float *GPU_upward,
                         float *CPU_update, float *CPU_upward,float *bias,
                         int *GPU_map,int *CPU_map, 
                         int numElements,int feature_size)
{
    int block_nodes = blockDim.y;
    int node_id = blockIdx.x * block_nodes + threadIdx.y;
    int idx = threadIdx.x;
    if(node_id < GPU_nodes)
    {
        for(int i = idx; i < feature_size; i += blockDim.x)
        {
            float tmp = GPU_update[node_id*feature_size+i] + GPU_upward[GPU_map[node_id]*feature_size+i] +CPU_upward[GPU_map[node_id]*feature_size+i]+bias[i];
            if(tmp< 0)
            {
                dst[GPU_map[node_id]*feature_size+i] = 0;
            }
            else
            {
                dst[GPU_map[node_id]*feature_size+i] = tmp;
            }
        }
    }
    else if(node_id < numElements)
    {
        node_id -= GPU_nodes;
        for(int i = idx; i < feature_size; i+= blockDim.x)
        {
            float tmp = CPU_update[node_id*feature_size+i] + GPU_upward[CPU_map[node_id]*feature_size+i] +CPU_upward[CPU_map[node_id]*feature_size+i]+bias[i];
            if(tmp< 0)
            {
                dst[CPU_map[node_id]*feature_size+i] = 0;
            }
            else
            {
                dst[CPU_map[node_id]*feature_size+i] = tmp;
            }
        }
    }

}
/**/
__global__ void addbiasrelu1(float *dst, int sample_size, 
                         float *GPU_update, float *GPU_upward,
                         float *CPU_update, float *CPU_upward, float *bias,
                         int *GPU_csr_indptr,int*CPU_csr_indptr,
                         int numElements,int feature_size)
{

    // int warp_x = blockDim.x / 32;
    // int block_warps = warp_x * blockDim.y;
    // int warp_id = threadIdx.y * warp_x + threadIdx.x /32;
    
    // int block_id = blockIdx.x + blockIdx.y *gridDim.x;
    // int idx = threadIdx.x % 32;
    
    int block_nodes = blockDim.y;
    int node_id = (blockIdx.y * gridDim.x + blockIdx.x) * block_nodes + threadIdx.y;
    int size_id = node_id % sample_size;
    int sampled_id = node_id / sample_size;
    int idx = threadIdx.x;
    if(sampled_id < numElements)
    {
        int GPU_start = GPU_csr_indptr[sampled_id];
        int GPU_nodes = GPU_csr_indptr[sampled_id+1] - GPU_start;
        int CPU_start = CPU_csr_indptr[sampled_id];
        int CPU_nodes = CPU_csr_indptr[sampled_id+1] - CPU_start;
        for(int j = idx; j < feature_size; j+=blockDim.x)
        {
            // printf("%d %d %d %d %d %d\n",blockIdx.x,blockDim.x,numElements,block_id,warp_id,idx);

            float tmp = GPU_upward[node_id*feature_size + j]+ CPU_upward[node_id*feature_size + j]+bias[j];
            if(size_id < GPU_nodes)
            {
                tmp += GPU_update[(GPU_start+size_id)*feature_size+j];
            }
            else 
            {
                tmp += CPU_update[(CPU_start+size_id-GPU_nodes)*feature_size+j];

            }
            if(tmp <  0)
            {

                dst[node_id*feature_size + j] = 0;
            }
            else dst[node_id*feature_size + j] = tmp;
        }

    }
}


/*
__global__ void addbiasrelu1(float *dst, int sample_size, 
                         float *GPU_update, float *GPU_upward,
                         float *CPU_update, float *CPU_upward, float *bias,
                         int *GPU_csr_indptr,int*CPU_csr_indptr,
                         int numElements,int feature_size)
{

    int warp_x = blockDim.x / 32;
    int block_warps = warp_x * blockDim.y;
    int warp_id = threadIdx.y * warp_x + threadIdx.x /32;
    
    int block_id = blockIdx.x + blockIdx.y *gridDim.x;
    int idx = threadIdx.x % 32;
   
    if(block_id < numElements)
    {
        int GPU_start = GPU_csr_indptr[block_id];
        int GPU_nodes = GPU_csr_indptr[block_id+1] - GPU_start;
        int CPU_start = CPU_csr_indptr[block_id];
        int CPU_nodes = CPU_csr_indptr[block_id+1] - CPU_start;
        for(int i = warp_id; i < sample_size; i+=block_warps)
        {
            for(int j = idx; j < feature_size; j+=32)
            {
                // printf("%d %d %d %d %d %d\n",blockIdx.x,blockDim.x,numElements,block_id,warp_id,idx);

                float tmp = GPU_upward[(block_id*sample_size + i)*feature_size + j]+ CPU_upward[(block_id*sample_size + i)*feature_size + j]+bias[j];
                if(i < GPU_nodes)
                {
                    tmp += GPU_update[(GPU_start+i)*feature_size+j];
                }
                else 
                {
                    tmp += CPU_update[(CPU_start+i-GPU_nodes)*feature_size+j];

                }
                if(tmp <  0)
                {

                    dst[(block_id*sample_size+i)*feature_size + j] = 0;
                }
                else dst[(block_id*sample_size+i)*feature_size + j] = tmp;
            }
        }

    }
}
*/
__global__ void addbiasrelu2(float *dst,int nElems,float *update,float *upward,float *bias,int feature_size)
{
    // int idx = threadIdx.x % 32;
    // int warp_x = blockDim.x /32;
    // int block_warps = warp_x * blockDim.y;
    // int warp_id = blockIdx.x * block_warps + threadIdx.y * warp_x + threadIdx.x /32;

    int block_nodes = blockDim.y;
    int node_id = blockIdx.x * block_nodes + threadIdx.y;
    int idx = threadIdx.x;

    if(node_id < nElems)
    {
        for(int i = idx; i < feature_size; i+=blockDim.x)
        {
            float tmp = update[node_id*feature_size + i] + upward[node_id*feature_size+i]+bias[i];
            if(tmp > 0)
            {
                dst[node_id*feature_size + i] = tmp;
            }
            else
            {
                dst[node_id*feature_size+i] = 0;
            }
        }
        
    }
}


__global__ void GPU_feature_transfer(float *dst,float *src,int32_t *id_map,int32_t * node_id,int num_nodes,int feature_size)
{
    // int warp_x = blockDim.x / 32;
    // int block_warps = warp_x * blockDim.y;
    // int warp_id = blockIdx.x * block_warps + threadIdx.y * warp_x + threadIdx.x /32;
    // int idx = threadIdx.x ;


    int block_nodes = blockDim.y;
    int node_idx = blockIdx.x * block_nodes + threadIdx.y;
    int idx = threadIdx.x;
    if(node_idx < num_nodes)
    {
        for(int i = idx; i < feature_size; i += blockDim.x)
        {
            dst[node_idx*feature_size+i] = src[id_map[node_id[node_idx]]*feature_size + i];

        }
    }
}
