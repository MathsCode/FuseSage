import torch

torch.ops.load_library('/home/eva_share/user_file/xujiaming/research/fuseGNN/fuseSagev5/build/libfuseSage.so')
import time

# generate Graph

nodes_num = 15
edges = 38
GPU_nodes = 5
intput_channels = 8
hidden_size = 10
output_channels = 5
num_layers = 2


# intput_channels = 100
# hidden_size = 256
# output_channels = 47
# num_layers = 2
# batch_size = 1024
# nodes_num = 

edge_csr_indptr = torch.zeros(1,nodes_num+1,dtype=torch.int32,device='cuda')
edge_csr_indices = torch.zeros(1,edges,dtype=torch.int32,device='cuda')
with open('./graph.txt','r') as f:
    for i,line in enumerate(f):
        u,v = eval(line.split(' ')[0]),eval(line.split(' ')[1])
        edge_csr_indices[0][i] = v-1
        edge_csr_indptr[0][u] = i+1

batch = torch.arange(GPU_nodes,dtype=torch.int32,device='cuda')
# edge_csr_indices = edge_csr_indices.cuda()
# edge_csr_indptr = edge_csr_indptr.cuda()

sample_size = torch.tensor([2,2],dtype=torch.int32)
sample_size_d = sample_size.cuda()

CPU_node_feature = torch.ones(nodes_num,intput_channels,dtype=torch.float32)*0.1

GPU_node_feature = torch.ones(GPU_nodes,intput_channels,dtype=torch.float32,device='cuda')*0.1
# GPU_node_feature = GPU_node_feature.cuda()

id2idx = torch.zeros(2,nodes_num,dtype=torch.int32)

for i in range(nodes_num):
    id2idx[0][i] = i
for i in range(GPU_nodes):
    id2idx[1][i] = 1
    id2idx[0][i] = i

id2idx_d = id2idx.cuda()

# print(CPU_node_feature)
# print(CPU_node_feature.shape)

# print(GPU_node_feature)
# print(GPU_node_feature.shape)

# print(id2idx)
# print(id2idx_d)

input_update_weights = (torch.ones(intput_channels,hidden_size,dtype=torch.float32,device='cuda'))
input_upward_weights = (torch.ones(intput_channels,hidden_size,dtype=torch.float32,device='cuda')*2)
input_upward_bias = (torch.ones(hidden_size,dtype=torch.float32,device='cuda')*3)

output_update_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*1)
output_upward_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*2)
output_upward_bias = (torch.ones(output_channels,dtype=torch.float32,device='cuda')*3)


hidden_update_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*1)
hidden_upward_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*2)
hidden_upward_bias = (torch.ones(output_channels,dtype=torch.float32,device='cuda')*3)


# print(batch)
# print(edge_csr_indptr)
# print(GPU_node_feature)
# print(id2idx_d)
s = time.time()
answer = torch.ops.fuseops.fuseSage(batch,edge_csr_indptr,edge_csr_indices,GPU_node_feature,CPU_node_feature,id2idx_d,id2idx,num_layers,sample_size,input_update_weights,input_upward_weights,input_upward_bias,output_update_weights,output_upward_weights,output_upward_bias,hidden_update_weights,hidden_upward_weights,hidden_upward_bias)
e = time.time()
print(e-s)
