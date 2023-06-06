#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

void update(float *feature, float *weights, float *update_results,
            int M, int K, int N, cudaStream_t stream);
void upward(float *update_results, int nnz,
            int32_t *csr_indptr_d, int32_t *csr_indice_d, float *csr_value_d, 
            float *upward_results, int M, int K, int N, cudaStream_t stream);