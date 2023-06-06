#include "../include/agg.h"

void update(float *feature,
            float *weights, 
            float *update_results,
            int M, 
            int K, 
            int N, 
            cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    float alpha = 1.f, beta = 0.f;
    cublasSgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                weights,
                N,
                feature,
                K,
                &beta,
                update_results,
                N);
}

void upward(float *update_results, int nnz,
            int32_t *csr_indptr_d, int32_t *csr_indice_d, float *csr_value_d,
            float *upward_results, int M, int K, int N, cudaStream_t stream)
{
    // Run cuSparseSpMM
    // Warning: upward need add bias
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSetStream(handle, stream);
    cusparseDnMatDescr_t UpdateDescr, UpwardDescr;
    cusparseSpMatDescr_t Adj;

    cusparseCreateDnMat(&UpdateDescr, K, N, N, update_results, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&UpwardDescr, M, N, N, upward_results, CUDA_R_32F, CUSPARSE_ORDER_ROW);


    cusparseCreateCsr(&Adj, M, K, nnz, csr_indptr_d, csr_indice_d, csr_value_d,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F);

    size_t workspace_size = 0;
    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, Adj, UpdateDescr, &beta,
                            UpwardDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                            &workspace_size);
    void *workspace = NULL;
    cudaMalloc(&workspace, workspace_size);
    cusparseSpMM(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                 CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                 &alpha, Adj, UpdateDescr, &beta, UpwardDescr,
                 CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
}