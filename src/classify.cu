#include "../include/classify.h"

__global__ void classify_kernel(int32_t *nodes,
                                const int num_nodes,
                                int32_t *node_device,
                                int32_t *GPU_node_buffer,
                                int32_t *CPU_node_buffer, int32_t *GPU_nodes, 
                                int32_t *CPU_nodes,
                                int32_t *GPU_map, 
                                int32_t *CPU_map)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes)
    {
        if (node_device[nodes[idx]] == 0)
        {
            int index = atomicAdd(CPU_nodes, 1);
            CPU_node_buffer[index] = nodes[idx];
            CPU_map[index] = idx;
        }
        else
        {
            int index = atomicAdd(GPU_nodes, 1);
            GPU_node_buffer[index] = nodes[idx];
            GPU_map[index] = idx;
        }
    }
}

void classify(int32_t *node_buf,
              int num_nodes,
              int32_t *node_device,
              int32_t* &GPU_node_buf,
              int32_t* &CPU_node_buf,
              int32_t* &device_nodes_number,
              int32_t* &GPU_map,
              int32_t* &CPU_map,
              int32_t *int_buf,
              cudaStream_t stream)
{
    GPU_map = int_buf;
    CPU_map = int_buf+num_nodes;
    GPU_node_buf = int_buf + 2*num_nodes;
    CPU_node_buf = int_buf + 3*num_nodes;
    device_nodes_number = int_buf+4*num_nodes;
    int threadsPerBlock = 1024;
    int blocksPerGrid  = (num_nodes+threadsPerBlock-1)/threadsPerBlock;
    classify_kernel<<<blocksPerGrid,threadsPerBlock,0,stream>>>(node_buf,num_nodes,node_device,GPU_node_buf,CPU_node_buf,device_nodes_number,device_nodes_number+1,GPU_map,CPU_map);
}