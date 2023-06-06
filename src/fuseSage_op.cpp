#include "utils.h"
#include "../include/fuseSage.h"
using torch::Tensor;

Tensor fuseSage_ext(Tensor &batch,
                    Tensor &edge_csr_indptr, Tensor &edge_csr_indice,
                    Tensor &GPU_feature, Tensor &CPU_feature,
                    Tensor &id2idx_d, Tensor &id2idx_h, int64_t num_layers,
                    Tensor &Sample_size_h,

                    Tensor &input_update_weights,
                    Tensor &input_upward_weights,
                    Tensor &input_upward_bias,
                    Tensor &output_update_weights,
                    Tensor &output_upward_weights,
                    Tensor &output_upward_bias,
                    Tensor &hidden_update_weights,
                    Tensor &hidden_upward_weights,
                    Tensor &hidden_upward_bias)
{

    
    // batch       : GPU
    // Edge_list   : GPU
    // GPU_feature : GPU
    // CPU_feature : CPU   pin_memory
    // id2idx      : GPU 0:CPU,1:GPU
    // Sample_size : CPU
    int32_t *batch_ptr = get_ptr<int32_t>(batch);
    int32_t *edge_csr_indptr_ptr = get_ptr<int32_t>(edge_csr_indptr);
    int32_t *edge_csr_indice_ptr = get_ptr<int32_t>(edge_csr_indice);
    float *GPU_feature_ptr = get_ptr<float>(GPU_feature);
    float *CPU_feature_ptr = get_ptr<float>(CPU_feature);
    int32_t *id2idx_d_ptr = get_ptr<int32_t>(id2idx_d);
    int32_t *id2idx_h_ptr = get_ptr<int32_t>(id2idx_h);
    int32_t *Sample_size_ptr = get_ptr<int32_t>(Sample_size_h);
    float *input_update_weights_ptr = get_ptr<float>(input_update_weights);
    float *input_upward_weights_ptr = get_ptr<float>(input_upward_weights);
    float *input_upward_bias_ptr = get_ptr<float>(input_upward_bias);

    float *output_update_weights_ptr = get_ptr<float>(output_update_weights);
    float *output_upward_weights_ptr = get_ptr<float>(output_upward_weights);
    float *output_upward_bias_ptr = get_ptr<float>(output_upward_bias);

    float *hidden_update_weights_ptr = nullptr;
    float *hidden_upward_weights_ptr = nullptr;
    float *hidden_upward_bias_ptr = nullptr;
    if (num_layers > 2)
    {
        hidden_update_weights_ptr = get_ptr<float>(hidden_update_weights);
        hidden_upward_weights_ptr = get_ptr<float>(hidden_upward_weights);
        hidden_upward_bias_ptr = get_ptr<float>(hidden_upward_bias);
    }

    auto input_weight_size = input_update_weights.sizes();
    //[input_channels,hiddensize]
    auto output_weight_size = output_update_weights.sizes();
    //[hidden_size,output_channels]
    auto id2idx_size = id2idx_h.sizes();

    auto batch_size = batch.sizes();
    // [node_nums]
    fuseSage(batch_ptr,
             edge_csr_indptr_ptr,
             edge_csr_indice_ptr,
             GPU_feature_ptr, 
             CPU_feature_ptr, 
             id2idx_d_ptr, 
             id2idx_h_ptr, 
             Sample_size_ptr, 
             input_update_weights_ptr,
             input_upward_weights_ptr,
             input_upward_bias_ptr,
             output_update_weights_ptr,
             output_upward_weights_ptr,
             output_upward_bias_ptr,
             hidden_update_weights_ptr,
             hidden_upward_weights_ptr,
             hidden_upward_bias_ptr,
             num_layers,id2idx_size[1],batch_size[0],
             input_weight_size[0],input_weight_size[1],output_weight_size[1],
             false,true
             );
    return batch; 
}

static auto registry = torch::RegisterOperators("fuseops::fuseSage", &fuseSage_ext);