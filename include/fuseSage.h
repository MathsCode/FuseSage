
void fuseSage(int32_t *batch_ptr,
              int32_t *edge_csr_indptr, int32_t *edge_csr_indice,
              float *GPU_feature_ptr, float *CPU_feature_ptr,
              int32_t *id2idx_ptr_d,int32_t *id2idx_ptr_h,
              int32_t *Sample_size_ptr, 
              float *input_update_weights_ptr,
              float *input_upward_weights_ptr,
              float *input_upward_bias_ptr,
              float *output_update_weights_ptr,
              float *output_upward_weights_ptr,
              float *output_bias_ptr,
              float *hidden_update_weights_ptr,
              float *hidden_upward_weights_ptr,
              float *hidden_bias_ptr,

              const int64_t num_layers, int node_nums,int batch_size,
              int input_channels,int hidden_size,const int output_channels,
              bool is_use_gcn = false,bool is_mixed = true);