#include "../include/fuseSage.h"
#include "../include/kernel.h"
#include "../include/sample.h"
#include "../include/classify.h"
#include "../include/agg.h"
#include "utils.h"
#include <cmath>
#include <time.h>
#define THREAD_PER_BLOCK 512
void fuseSage(int32_t *batch_ptr,
              int32_t *edge_csr_indptr, 
              int32_t *edge_csr_indice,
              float *GPU_feature_ptr, 
              float *CPU_feature_ptr,
              int32_t *id2idx_ptr_d,
              int32_t *id2idx_ptr_h,
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
              int input_channels,int hidden_size,int output_channels,
              bool is_use_gcn,bool is_mixed)
{


    // optimize1: memory malloc merge
    // optimize2: upward_update && update parallel
    // optimize3: memory reuse

    int bias = 10;
    cudaStream_t sample_stream,add_stream,classify_stream;

    int32_t* node_device = id2idx_ptr_d + node_nums;
    cudaStreamCreate(&sample_stream);
    cudaStreamCreate(&add_stream);
    cudaStreamCreate(&classify_stream);
    cudaStream_t CPUstream[num_layers+1],GPUstream[num_layers+1];
    cudaEvent_t CPUevent[num_layers+1],GPUevent[num_layers+1],classifyEvent[(num_layers+1)*num_layers/2];

    for(int i = 0; i <= num_layers;i++)
    {
        cudaEventCreate(&GPUevent[i]);
        cudaEventCreate(&CPUevent[i]);
        cudaStreamCreate(&GPUstream[i]);
        cudaStreamCreate(&CPUstream[i]);
    }
    for(int i = 0; i < (num_layers+1)*num_layers/2; i++)
    {
        cudaEventCreate(&classifyEvent[i]);
    }
    
    
    
    if(is_mixed)
    {
        
        int32_t *GPU_node_buffer[num_layers+1];
        int32_t *CPU_node_buffer[num_layers+1];
        int32_t *device_node_buffer[num_layers+1];
        // memory: sum(num_nodes[1]~num_nodes[num_layers]) + num_nodes[0]*2
        
        int32_t *node_buffer[num_layers+1];
        // memory: sum(num_nodes[1]~num_nodes[num_layers])
        
        float* GPU_feature_buffer[num_layers+1];
        float* CPU_feature_buffer[num_layers+1];
        float* device_feature_buffer[num_layers+1];
        // memory: sum(num_nodes[0]~num_nodes[num_layers])*input_channels
        

        float* result_feature_buffer[num_layers];
        // memory: sum(num_nodes[0]~num_nodes[num_layers-1])*max(hidden_size,output_channels)
        
        float* final_update_results = nullptr;
        // memory: num_nodes[0]*output_channels
        
        float* final_upward_results = nullptr;
        // memory: num_nodes[0]*output_channels
        
        
        float* final_upward_cache = nullptr;
        // memory: num_nodes[1]*output_channels
        
        int GPU_nodes_number[num_layers+1]={0};
        int CPU_nodes_number[num_layers+1]={0};
        int num_nodes[num_layers+1] = {0}; 

        int32_t *device_nodes_number = nullptr;
        // memory: 2
        
        int32_t *csrallindptr[num_layers],*csrallindices[num_layers];
        // memory: sum(num_nodes[0]~num_nodes[num_layers-1])+num_layers+sum(num_nodes[1]~num_nodes[num_layers])
        
        
        float *csrallval[num_layers];
        // memory: sum(num_nodes[1]~num_nodes[num_layers])
        

        int32_t *csrGPUindices[num_layers];
        int32_t *csrCPUindices[num_layers];
        int32_t *device_csr_indices[num_layers];
        // memory:sum(num_nodes[1]~num_nodes[num_layers])
        
        float *csrGPUval[num_layers],*csrCPUval[num_layers];
        float *device_csr_value[num_layers];
        // memory:sum(num_nodes[1]~num_nodes[num_layers])
        
        thrust::device_vector<int> GPU_nodes_pernode[num_layers];
        thrust::device_vector<int> CPU_nodes_pernode[num_layers];

        float *CPU_update_results[num_layers];
        float *GPU_update_results[num_layers];
        float *device_update_results[num_layers];
        // memory:sum(num_nodes[0]~num_nodes[num_layers-1])*hidden_size
        

        float *GPU_upward_cache[num_layers+1];
        float *CPU_upward_cache[num_layers+1];
        float *device_upward_cache[num_layers+1];
        // memory:sum(num_nodes[1]~num_nodes[num_layers])*hidden_size
        

        float *CPU_upward_results[num_layers+1];
        float *GPU_upward_results[num_layers+1];
        float *device_upward_results[num_layers+1];
        // memory:sum(num_nodes[0]~num_nodes[num_layers-1])*hidden_size

        
        int32_t *GPU_map = nullptr,*CPU_map = nullptr;
        // memory: num_nodes[0]*2

        num_nodes[0] = batch_size;
        int tot_num = num_nodes[0];
        node_buffer[0] = batch_ptr;
        for(int i = 1; i <= num_layers; i++)
        {
            num_nodes[i] = num_nodes[i-1] *Sample_size_ptr[i-1];
            tot_num += num_nodes[i];
        }

        int tot_float_buf = (tot_num*input_channels)+(tot_num-num_nodes[num_layers])*max(hidden_size,output_channels) + num_nodes[0]*output_channels*2 + num_nodes[1]*output_channels + (tot_num - num_nodes[0])*2+(tot_num-num_nodes[num_layers])*hidden_size*2+(tot_num-num_nodes[0])*hidden_size;
        
        int tot_int_buf = tot_num*2+ 2 + tot_num*2-num_nodes[num_layers]+num_layers+ tot_num;

        int32_t *int_buf = nullptr;
        float *float_buf = nullptr;
        cudaMalloc((void **)&int_buf,tot_int_buf*sizeof(int32_t));
        cudaMalloc((void **)&float_buf,tot_float_buf*sizeof(float));
        printf("int:%d float:%d \n",tot_int_buf,tot_float_buf);

        int int_alloced = 0;
        int float_alloced = 0;
        

        dim3 block(256,4,1);
        int block_warps = 32;
        
        clock_t start_time,end_time;
        start_time = clock();
        for(int layer = 0; layer <= num_layers; layer++)
        {

            // arange tasks in sample stream
            printf("layer %d start......\n",layer);

            // node_buffer memory
            if(layer < num_layers)
            {

                GPU_nodes_pernode[layer].resize(num_nodes[layer]+1);
                CPU_nodes_pernode[layer].resize(num_nodes[layer]+1);

                // std::cout<<num_nodes[layer]<<"\n";
                sample(node_buffer[layer],num_nodes[layer],node_buffer[layer+1],num_nodes[layer+1],Sample_size_ptr[layer],node_device,edge_csr_indptr,edge_csr_indice,thrust::raw_pointer_cast(GPU_nodes_pernode[layer].data()),thrust::raw_pointer_cast(CPU_nodes_pernode[layer].data()),csrallindptr[layer],csrallindices[layer],csrallval[layer],int_buf+int_alloced,float_buf+float_alloced,sample_stream);

                int_alloced += num_nodes[layer+1]*2+num_nodes[layer]+1;
                float_alloced += num_nodes[layer+1];

                // // step 7
                thrust::exclusive_scan(thrust::cuda::par.on(sample_stream),GPU_nodes_pernode[layer].begin(),GPU_nodes_pernode[layer].end(),GPU_nodes_pernode[layer].begin());
                thrust::exclusive_scan(thrust::cuda::par.on(sample_stream),CPU_nodes_pernode[layer].begin(),CPU_nodes_pernode[layer].end(),CPU_nodes_pernode[layer].begin());

                // cudaStreamSynchronize(sample_stream);
                // printf("nodes in layer%d,number:%d\n",layer,num_nodes[layer]);
                // print_check<int32_t>(node_buffer[layer],num_nodes[layer]);
                // printf("GPU nodes per node\n");
                // print_check<int32_t>(thrust::raw_pointer_cast(GPU_nodes_pernode[layer].data()),num_nodes[layer]+1);
                // printf("CPU nodes per node\n");
                // print_check<int32_t>(thrust::raw_pointer_cast(CPU_nodes_pernode[layer].data()),num_nodes[layer]+1);
                // printf("Sample nodes in layer%d,number:%d\n",layer,num_nodes[layer+1]);
                // print_check<int32_t>(node_buffer[layer+1],num_nodes[layer+1]);
                
            }
        
            
            // arange tasks in classify stream
            if(layer == 0)
            {
                //step 1
                //pick CPU node and GPU node 
                // use default stream
                classify(node_buffer[0],num_nodes[0],node_device,GPU_node_buffer[0],CPU_node_buffer[0],device_nodes_number,GPU_map,CPU_map,int_buf+int_alloced,classify_stream);
                int_alloced += 4*num_nodes[0]+2;

                
                cudaMemcpyAsync(GPU_nodes_number,device_nodes_number,sizeof(int32_t),cudaMemcpyDeviceToHost,classify_stream);
                cudaMemcpyAsync(CPU_nodes_number,device_nodes_number+1,sizeof(int32_t),cudaMemcpyDeviceToHost,classify_stream);
                cudaStreamSynchronize(classify_stream);

                // printf("GPU nodes in layer%d,number:%d\n",layer,GPU_nodes_number[layer]);
                // print_check<int32_t>(GPU_node_buffer[layer],GPU_nodes_number[layer]);
                // printf("CPU nodes in layer%d,number:%d\n",layer,CPU_nodes_number[layer]);
                // print_check<int32_t>(CPU_node_buffer[layer],CPU_nodes_number[layer]);
            }

           
            // optimize TODO
            if(layer < num_layers)
            {
                result_feature_buffer[layer] = float_buf + float_alloced;
                float_alloced += num_nodes[layer]*max(output_channels,hidden_size);
            }

            
            //step 3
            //get GPU feature
            GPU_feature_buffer[layer] = float_buf+float_alloced;
            float_alloced += GPU_nodes_number[layer]*input_channels;

            int block_x = min(32*(input_channels/32+1),THREAD_PER_BLOCK);
            int block_y = THREAD_PER_BLOCK/block_x;
            int grid_x = (GPU_nodes_number[layer]+block_y-1)/block_y;
            dim3 trans_block(block_x,block_y,1);   // 8 warps;
            dim3 trans_grid(grid_x,1,1);
            GPU_feature_transfer<<<trans_grid,trans_block,0,GPUstream[layer]>>>(GPU_feature_buffer[layer],GPU_feature_ptr,id2idx_ptr_d,GPU_node_buffer[layer],GPU_nodes_number[layer],input_channels);
            
            // dim3 tranfer_grid((GPU_nodes_number[layer]+block_warps-1)/block_warps,1,1);
            // GPU_feature_transfer<<<tranfer_grid,block,0,GPUstream[layer]>>>(GPU_feature_buffer[layer],GPU_feature_ptr,id2idx_ptr_d,GPU_node_buffer[layer],GPU_nodes_number[layer],input_channels);

            
            //step 4
            //get CPU feature  
            int32_t *CPU_node_buffer_h = nullptr;
            CPU_node_buffer_h = (int32_t *)malloc((CPU_nodes_number[layer])*sizeof(int32_t));

            cudaMemcpyAsync(CPU_node_buffer_h,CPU_node_buffer[layer],(CPU_nodes_number[layer])*sizeof(int32_t),cudaMemcpyDeviceToHost,CPUstream[layer]);

            CPU_feature_buffer[layer] = float_buf + float_alloced;
            float_alloced += CPU_nodes_number[layer]*input_channels;
            
            for(int i = 0; i < CPU_nodes_number[layer]; i++)
            {
                cudaMemcpyAsync(CPU_feature_buffer[layer]+i*input_channels,CPU_feature_ptr+id2idx_ptr_h[CPU_node_buffer_h[i]]*input_channels,input_channels*sizeof(float),cudaMemcpyHostToDevice,CPUstream[layer]);
            }
            // cudaStreamSynchronize(GPUstream[layer]);
            // printf("GPU feature in layer%d,number:%d\n",layer,GPU_nodes_number[layer]);
            // print_check<float>(GPU_feature_buffer[layer],GPU_nodes_number[layer]*input_channels);

            // cudaStreamSynchronize(CPUstream[layer]);
            // printf("CPU feature in layer%d,number:%d\n",layer,CPU_nodes_number[layer]);
            // print_check<float>(CPU_feature_buffer[layer],CPU_nodes_number[layer]*input_channels);
            
            //step 5
            // GPU feature aggregate
            
            
            int M_GPU = GPU_nodes_number[layer];
            int M_CPU = CPU_nodes_number[layer];
            int K = input_channels;
            int N = hidden_size;
            
            
            if(layer < num_layers)
            {
                CPU_update_results[layer] = float_buf + float_alloced;
                float_alloced += M_CPU*N;
                GPU_update_results[layer] = float_buf + float_alloced;
                float_alloced += M_GPU*N;
            }
            if(layer > 0)
            {
                CPU_upward_cache[layer] = float_buf + float_alloced;
                float_alloced += M_CPU*N;
                GPU_upward_cache[layer] = float_buf + float_alloced;
                float_alloced += M_GPU*N;
                CPU_upward_results[layer] = float_buf + float_alloced;
                float_alloced += num_nodes[layer-1]*N;
                GPU_upward_results[layer] = float_buf + float_alloced;
                float_alloced += num_nodes[layer-1]*N;
            }
            

            // step 5
            if(layer < num_layers)
            {
                update(GPU_feature_buffer[layer],input_update_weights_ptr,GPU_update_results[layer],M_GPU,K,N,GPUstream[layer]);
            
                update(CPU_feature_buffer[layer],input_update_weights_ptr,CPU_update_results[layer],M_CPU,K,N,CPUstream[layer]);
            }
            

            
            
            
            
            if(layer > 0)
            {
                
                update(GPU_feature_buffer[layer],input_upward_weights_ptr,GPU_upward_cache[layer],M_GPU,K,N,GPUstream[layer]);
            
                update(CPU_feature_buffer[layer],input_upward_weights_ptr,CPU_upward_cache[layer],M_CPU,K,N,CPUstream[layer]);
                
                upward(GPU_upward_cache[layer],GPU_nodes_number[layer],thrust::raw_pointer_cast(GPU_nodes_pernode[layer-1].data()),csrGPUindices[layer-1],csrGPUval[layer-1],GPU_upward_results[layer],num_nodes[layer-1],M_GPU,N,GPUstream[layer]);

                upward(CPU_upward_cache[layer],CPU_nodes_number[layer],thrust::raw_pointer_cast(CPU_nodes_pernode[layer-1].data()),csrCPUindices[layer-1],csrCPUval[layer-1],CPU_upward_results[layer],num_nodes[layer-1],M_CPU,N,CPUstream[layer]);

                // cudaStreamSynchronize(GPUstream[layer]);
                // printf("GPU_upward_results in layer%d,number:%d\n",layer,GPU_nodes_number[layer]);
                // print_check<float>(GPU_upward_results[layer],num_nodes[layer-1]*N);

                // cudaStreamSynchronize(CPUstream[layer]);
                // printf("CPU_upward_results in layer%d,number:%d\n",layer,CPU_nodes_number[layer]);
                // print_check<float>(CPU_upward_results[layer],num_nodes[layer-1]*N);
                
            }
            
            cudaEventRecord(CPUevent[layer],CPUstream[layer]);
            cudaEventRecord(GPUevent[layer],GPUstream[layer]);
            
            
            if(layer == 1)
            {
                
                cudaStreamWaitEvent(add_stream,CPUevent[layer-1]);
                cudaStreamWaitEvent(add_stream,CPUevent[layer]);
                cudaStreamWaitEvent(add_stream,GPUevent[layer-1]);
                cudaStreamWaitEvent(add_stream,GPUevent[layer]);
                int block_x = min(32*(N/32+1),THREAD_PER_BLOCK);
                int block_y = THREAD_PER_BLOCK/block_x;
                int grid_x = (num_nodes[layer-1]+block_y-1)/block_y;
                dim3 add_block(block_x,block_y,1);   // 8 warps;
                dim3 add_grid(grid_x,1,1);
                addbiasrelu0<<<add_grid,add_block,0,add_stream>>>(result_feature_buffer[layer-1],GPU_nodes_number[layer-1],GPU_update_results[layer-1],GPU_upward_results[layer],CPU_update_results[layer -1],CPU_upward_results[layer],input_upward_bias_ptr,GPU_map,CPU_map,num_nodes[layer-1],N);
                // cudaStreamSynchronize(add_stream);
                // printf("result_feature_buffer layer%d,number:%d\n",layer-1,num_nodes[layer-1]);
                // print_check<float>(result_feature_buffer[layer-1],num_nodes[layer-1]*hidden_size);
                // free GPU_map,CPU_map update_result upward_result csrGPU csrCPU
                
            }
            
            if(layer > 1)
            {
                cudaStreamWaitEvent(add_stream,CPUevent[layer-1]);
                cudaStreamWaitEvent(add_stream,CPUevent[layer]);
                cudaStreamWaitEvent(add_stream,GPUevent[layer-1]);
                cudaStreamWaitEvent(add_stream,GPUevent[layer]);
                // dim3 add_block(128,2,1);   // 8 warps;
                int block_x = min(32*(N/32+1),THREAD_PER_BLOCK);
                int block_y = THREAD_PER_BLOCK/block_x;
                int grid_x = (num_nodes[layer-1]+block_y-1)/block_y;
                dim3 add_block(block_x,block_y,1);
                dim3 add_grid(grid_x,1,1);
                addbiasrelu1<<<add_grid,add_block,0,add_stream>>>(result_feature_buffer[layer-1],Sample_size_ptr[layer-2],GPU_update_results[layer-1],GPU_upward_results[layer],CPU_update_results[layer-1],CPU_upward_results[layer],input_upward_bias_ptr,thrust::raw_pointer_cast(GPU_nodes_pernode[layer-2].data()),thrust::raw_pointer_cast(CPU_nodes_pernode[layer-2].data()),num_nodes[layer-2],N);

                // printf("GPU_update_results layer%d,number:%d\n",layer-1,GPU_nodes_number[layer-1]);
                // print_check<float>(GPU_update_results[layer-1],GPU_nodes_number[layer-1]*hidden_size);
                // printf("GPU_upward_results layer%d,number:%d\n",layer,num_nodes[layer-1]);
                // print_check<float>(GPU_upward_results[layer],num_nodes[layer-1]*hidden_size);


                // printf("CPU_update_results layer%d,number:%d\n",layer-1,CPU_nodes_number[layer-1]);
                // print_check<float>(CPU_update_results[layer-1],CPU_nodes_number[layer-1]*hidden_size);
                // printf("CPU_upward_results layer%d,number:%d\n",layer,num_nodes[layer-1]);
                // print_check<float>(CPU_upward_results[layer],num_nodes[layer-1]*hidden_size);
                // cudaEventDestroy(classifyEvent[layer-1]);
                // cudaStreamSynchronize(add_stream);
                // printf("result_feature_buffer layer%d,number:%d\n",layer-1,num_nodes[layer-1]);
                // print_check<float>(result_feature_buffer[layer-1],num_nodes[layer-1]*hidden_size);
            }
            // cudaStreamSynchronize(add_stream);
            if(layer <num_layers)
            {
                cudaStreamSynchronize(sample_stream);
                
                GPU_nodes_number[layer+1] = *(GPU_nodes_pernode[layer].end()-1);
                CPU_nodes_number[layer+1] = *(CPU_nodes_pernode[layer].end()-1);
                // printf("GPU_nodes_number in layer %d %d\n",layer+1,GPU_nodes_number[layer+1]);
                // printf("CPU_nodes_number in layer %d %d\n",layer+1,CPU_nodes_number[layer+1]);
                // csrGPUindices
                get_csr(num_nodes[layer],node_buffer[layer+1],Sample_size_ptr[layer],thrust::raw_pointer_cast(GPU_nodes_pernode[layer].data()),thrust::raw_pointer_cast(CPU_nodes_pernode[layer].data()),csrGPUindices[layer],csrCPUindices[layer],csrGPUval[layer],csrCPUval[layer],GPU_nodes_number[layer+1],CPU_nodes_number[layer+1],GPU_node_buffer[layer+1],CPU_node_buffer[layer+1],int_buf+int_alloced,float_buf+float_alloced,sample_stream);
                int_alloced += (GPU_nodes_number[layer+1] + CPU_nodes_number[layer+1])*2;
                float_alloced += GPU_nodes_number[layer+1] + CPU_nodes_number[layer+1];

                // printf("nodes in layer%d,number:%d\n",layer,num_nodes[layer]);
                // print_check<int32_t>(node_buffer[layer],num_nodes[layer]);
                // printf("GPU nodes per node\n");
                // print_check<int32_t>(thrust::raw_pointer_cast(GPU_nodes_pernode[layer].data()),num_nodes[layer]+1);
                // printf("CPU nodes per node\n");
                // print_check<int32_t>(thrust::raw_pointer_cast(CPU_nodes_pernode[layer].data()),num_nodes[layer]+1);
                // printf("Sample nodes in layer%d,number:%d\n",layer,num_nodes[layer+1]);
                // print_check<int32_t>(node_buffer[layer+1],num_nodes[layer+1]);
                // printf("csrGPUindices layer%d,number:%d\n",layer,GPU_nodes_number[layer+1]);
                // print_check<int32_t>(csrGPUindices[layer],GPU_nodes_number[layer+1]);
                // printf("csrCPUindices layer%d,number:%d\n",layer,CPU_nodes_number[layer+1]);
                // print_check<int32_t>(csrCPUindices[layer],CPU_nodes_number[layer+1]);
                // printf("GPU_node_buffer layer%d,number:%d\n",layer+1,GPU_nodes_number[layer+1]);
                // print_check<int32_t>(GPU_node_buffer[layer+1],GPU_nodes_number[layer+1]);
                // printf("CPU_node_buffer layer%d,number:%d\n",layer+1,CPU_nodes_number[layer+1]);
                // print_check<int32_t>(CPU_node_buffer[layer+1],CPU_nodes_number[layer+1]);
            }
            
            
        }
        
        
        cudaStreamSynchronize(add_stream);
        final_update_results = float_buf + float_alloced;
        float_alloced += num_nodes[0]*output_channels;
        

        update(result_feature_buffer[0],output_update_weights_ptr,final_update_results,num_nodes[0],hidden_size,output_channels,GPUstream[0]);
        cudaEventRecord(GPUevent[0]);

        // cudaStreamSynchronize(GPUstream[0]);
        // printf("result_feature_buffer layer%d,number:%d\n",0,num_nodes[0]);
        // print_check<float>(result_feature_buffer[0],num_nodes[0]*output_channels);

        // printf("update_feature_results layer%d,number:%d\n",0,num_nodes[0]);
        // print_check<float>(update_feature_results[0],num_nodes[0]*output_channels);

        final_upward_cache = float_buf + float_alloced;
        float_alloced +=num_nodes[1]*output_channels; 
        final_upward_results = float_buf + float_alloced;
        float_alloced += num_nodes[0]*output_channels;

        update(result_feature_buffer[1],output_upward_weights_ptr,final_upward_cache,num_nodes[1],hidden_size,output_channels,GPUstream[1]);
        upward(final_upward_cache,num_nodes[1],csrallindptr[0],csrallindices[0],csrallval[0],final_upward_results,num_nodes[0],num_nodes[1],output_channels,GPUstream[1]);

        cudaEventRecord(GPUevent[1]);
        cudaStreamWaitEvent(add_stream,GPUevent[0]);
        cudaStreamWaitEvent(add_stream,GPUevent[1]);
        // cudaStreamSynchronize(GPUstream[1]);
        // printf("result_feature_buffer layer%d,number:%d\n",0,num_nodes[0]);
        // print_check<float>(result_feature_buffer[1],num_nodes[1]*output_channels);

        // printf("upward_feature_cache layer%d,number:%d\n",0,num_nodes[1]);
        // print_check<float>(upward_feature_cache[1],num_nodes[1]*output_channels);

        // printf("upward_feature_results layer%d,number:%d\n",0,num_nodes[0]);
        // print_check<float>(upward_feature_results[1],num_nodes[0]*output_channels);
        int block_x = min(32*(output_channels/32+1),THREAD_PER_BLOCK);
        int block_y = THREAD_PER_BLOCK/block_x;
        int grid_x = (num_nodes[0]+block_y-1)/block_y;
        dim3 add_block(block_x,block_y,1);   
        dim3 add_grid(grid_x,1,1);
        // dim3 grid((num_nodes[0]+block_warps-1)/block_warps,1,1);
        addbiasrelu2<<<add_grid,add_block,0,add_stream>>>(result_feature_buffer[0],num_nodes[0],final_update_results,final_upward_results,output_bias_ptr,output_channels);
        /**/
        end_time = clock();
        printf("int allocated: %d,float allocated: %d",int_alloced,float_alloced);
        printf("Run time: %lf\n",(double)(end_time - start_time)/2/CLOCKS_PER_SEC);
        
        
        cudaDeviceSynchronize();
        cudaFree(int_buf);
        cudaFree(float_buf);
        // printf("result_feature_buffer layer%d,number:%d\n",0,num_nodes[0]);
        // print_check<float>(result_feature_buffer[0],num_nodes[0]*output_channels); 
        
    }

    
}
