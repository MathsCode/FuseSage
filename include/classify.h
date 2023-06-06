#include <cuda_runtime.h>
#include <cublas_v2.h>

void classify(int32_t *node_buf,
              int num_nodes,
              int32_t *node_device,
              int32_t* &GPU_node_buf,
              int32_t* &CPU_node_buf,
              int32_t* &device_nodes_number,
              int32_t* &GPU_map,
              int32_t* &CPU_map,
              int32_t *int_buf,
              cudaStream_t stream);