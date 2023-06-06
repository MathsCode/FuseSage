// #pragma once
#include "../include/sample.h"
#include "./utils.h"
__global__ void sample_kernel(int32_t *node_buffer,
                              int32_t num_nodes,
                              int sample_size,
                              int32_t *node_device,
                              int32_t *edge_csr_indptr, 
                              int32_t *edge_csr_indices,
                              int32_t *Sample_node_buffer, 
                              int *GPU_nodes_pernode,
                              int *CPU_nodes_pernode,
                              int32_t *csr_all_indptr, int32_t *csr_all_indices,
                              float *csr_all_val)
{
    int node_x = blockDim.x / 16;
    int block_nodes = node_x * blockDim.y;
    int node_id = blockIdx.x * block_nodes + threadIdx.y * node_x + threadIdx.x / 16;
    int idx = threadIdx.x % 16;
    if (node_id < num_nodes)
    {

        csr_all_indptr[node_id] = node_id * sample_size;
        int32_t neighbour_start = edge_csr_indptr[node_buffer[node_id]];
        int32_t neighbour_end = edge_csr_indptr[node_buffer[node_id] + 1];
        
        for (int i = idx; i < sample_size; i += 16)
        {
            
            int sample_node_id = edge_csr_indices[neighbour_start + i % (neighbour_end - neighbour_start)];
            // printf("%d \n",sample_node_id);
            if (node_device[sample_node_id] == 1)
            {
                // printf("%d %d %d %d %d\n",warp_x,block_warps,threadIdx.y,warp_id,num_nodes);
                int index = atomicAdd(&GPU_nodes_pernode[node_id], 1);
                Sample_node_buffer[node_id * sample_size + index] = sample_node_id;
            }
            else
            {
                int index = atomicAdd(&CPU_nodes_pernode[node_id], 1);
                Sample_node_buffer[node_id * sample_size + sample_size - 1 - index] = sample_node_id;
            }
            csr_all_indices[node_id * sample_size + i] = node_id * sample_size + i;

            csr_all_val[node_id * sample_size + i] = 1.0;
        }
    }
}

__global__ void get_device_csr(int num_nodes, int32_t *sample_node_buffer, int sample_size,
                               int32_t *GPU_csr_indptr, int32_t *CPU_csr_indptr,
                               int32_t *GPU_csr_indices, int32_t *CPU_csr_indices,
                               float * GPU_csr_val,float * CPU_csr_val,
                               int32_t *GPU_buffer_nodes, int32_t *CPU_buffer_nodes)
{
    int node_x = blockDim.x / 16;
    int block_nodes = node_x * blockDim.y;
    int node_id = blockIdx.x * block_nodes + threadIdx.y * node_x + threadIdx.x / 16;
    int idx = threadIdx.x % 16;
    if (node_id < num_nodes)
    {
        int tot_GPU = (GPU_csr_indptr[node_id + 1] - GPU_csr_indptr[node_id]);
        int tot_CPU = (CPU_csr_indptr[node_id + 1] - CPU_csr_indptr[node_id]);
        /*
        for (int i = idx; i < max(tot_CPU, tot_GPU); i += 32)
        {
            if (i < tot_GPU)
            {
                GPU_buffer_nodes[GPU_csr_indptr[warp_id] + i] = sample_node_buffer[warp_id * sample_size + i];
                GPU_csr_indices[GPU_csr_indptr[warp_id] + i] = GPU_csr_indptr[warp_id] + i;
                GPU_csr_val[GPU_csr_indptr[warp_id] + i] = 1.0;
            }
            if (i < tot_CPU)
            {
                CPU_buffer_nodes[CPU_csr_indptr[warp_id] + i] = sample_node_buffer[warp_id * sample_size + tot_GPU+i];
                CPU_csr_indices[CPU_csr_indptr[warp_id] + i] = CPU_csr_indptr[warp_id] + i;
                CPU_csr_val[CPU_csr_indptr[warp_id] + i] = 1.0;

            }
        }
        */
        for (int i = idx; i < sample_size; i += 16)
        {
            if(i < tot_GPU)
            {
                GPU_buffer_nodes[GPU_csr_indptr[node_id] + i] = sample_node_buffer[node_id * sample_size + i];
                GPU_csr_indices[GPU_csr_indptr[node_id] + i] = GPU_csr_indptr[node_id] + i;
                GPU_csr_val[GPU_csr_indptr[node_id] + i] = 1.0;
            }
            else
            {
                int id = i - tot_GPU;
                CPU_buffer_nodes[CPU_csr_indptr[node_id] + id ] = sample_node_buffer[node_id * sample_size + tot_GPU+id];
                CPU_csr_indices[CPU_csr_indptr[node_id] + id] = CPU_csr_indptr[node_id] + id;
                CPU_csr_val[CPU_csr_indptr[node_id] + id] = 1.0;
            }
        }
    }
}

// int cal_buf(int cur_num_nodes,int sample_size)
// {

// }
void sample(int32_t* cur_node_buf,
            int cur_num_nodes,
            int32_t* &next_node_buf,
            int next_num_nodes,
            int sample_size,
            int32_t *node_device,
            int32_t *edge_csr_indptr,
            int32_t *edge_csr_indices,
            int* GPU_csr_indptr,
            int* CPU_csr_indptr,
            int32_t* &ALL_csr_indptr,
            int32_t* &ALL_csr_indices,
            float* &ALL_csr_value,
            int32_t *int_buf,
            float   *float_buf,
            cudaStream_t stream)
{
    next_node_buf = int_buf;
    ALL_csr_indptr = int_buf + next_num_nodes;
    ALL_csr_indices = int_buf + next_num_nodes+ cur_num_nodes + 1;
    ALL_csr_value = float_buf;
    // 16 threads for sampling a node
    dim3 block(256, 4, 1);
    int block_n = block.x*block.y/16;
    dim3 grid((cur_num_nodes+block_n-1)/block_n,1,1);

    sample_kernel<<<grid, block, 0, stream>>>(cur_node_buf, cur_num_nodes, sample_size, node_device, edge_csr_indptr, edge_csr_indices, next_node_buf, GPU_csr_indptr, CPU_csr_indptr, ALL_csr_indptr, ALL_csr_indices, ALL_csr_value);
}

void get_csr(int cur_num_nodes,
             int32_t *next_node_buf, 
             int sample_size,
             int32_t* GPU_csr_indptr,
             int32_t* CPU_csr_indptr,
             int32_t* &GPU_csr_indices,
             int32_t* &CPU_csr_indices,
             float* &GPU_csr_value,
             float* &CPU_csr_value,
             int next_GPU_nodes_num,
             int next_CPU_nodes_num,
             int32_t* &next_GPU_node_buf,
             int32_t* &next_CPU_node_buf,
             int32_t *int_buf,
             float *float_buf,
             cudaStream_t stream)
{
    GPU_csr_indices = int_buf;
    GPU_csr_value = float_buf;
    CPU_csr_indices = int_buf+next_GPU_nodes_num;
    CPU_csr_value = float_buf+next_CPU_nodes_num;
    next_GPU_node_buf = int_buf + next_GPU_nodes_num+next_CPU_nodes_num;
    next_CPU_node_buf = int_buf + next_GPU_nodes_num*2+next_CPU_nodes_num;
    
    dim3 block(256,4,1);
    int block_n = block.x*block.y/16;
    dim3 grid((cur_num_nodes+block_n-1)/block_n,1,1);
    get_device_csr<<<grid,block,0,stream>>>(cur_num_nodes,next_node_buf,sample_size,GPU_csr_indptr,CPU_csr_indptr,GPU_csr_indices,CPU_csr_indices,GPU_csr_value,CPU_csr_value,next_GPU_node_buf,next_CPU_node_buf);
    // cudaStreamSynchronize(stream);

}