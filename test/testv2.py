import argparse

import dgl
import dgl.nn as dglnn
from dgl.nn.pytorch import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset
import numpy as np
dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root='/user/MathsCode/models/GNN/dataset/DGL'))

g = dataset[0]

edge_csr = g.adj().to_sparse_csr()

edge_csr_indptr = edge_csr.crow_indices()
edge_csr_indices = edge_csr.col_indices()
npy = np.zeros(edge_csr_indptr.shape[0])
# for i in range(edge_csr_indptr.shape[0]-1):
#     npy[i] = edge_csr_indptr[i+1] - edge_csr_indptr[i]
dct = {}
for i in range(edge_csr_indptr.shape[0]-1):
    dct[i] = edge_csr_indptr[i+1] - edge_csr_indptr[i]

dct_sort = sorted(dct.items(),key=lambda x:x[1], reverse=True)
feats = g.ndata['feat']
nodes_num = feats.shape[0]
GPU_nodes = feats.shape[0]//2
intput_channels = 100
hidden_size = 256
output_channels = 47
num_layers = 2
batch_size = 1024
idx2id = torch.zeros(2,nodes_num,dtype=torch.int32)

for i in range(feats.shape[0]):
    idx2id[0][i] = i
for i in range(feats.shape[0]//2):
    idx2id[0][dct_sort[i][0]] = i
    idx2id[1][dct_sort[i][0]] = 1
idx2id_d = idx2id.cuda()

sample_size = torch.tensor([5,10],dtype=torch.int32)

sample_size_d = sample_size.cuda()


CPU_node_feature = torch.ones(nodes_num,intput_channels,dtype=torch.float32)*0.1
GPU_node_feature = torch.ones(GPU_nodes,intput_channels,dtype=torch.float32,device='cuda')*0.1
batch = torch.zeros(batch_size,dtype=torch.int32,device='cuda')
for i in range(batch_size):
    batch[i] = dct_sort[i][0]


input_update_weights = (torch.ones(intput_channels,hidden_size,dtype=torch.float32,device='cuda'))
input_upward_weights = (torch.ones(intput_channels,hidden_size,dtype=torch.float32,device='cuda')*(-0.2))
input_upward_bias = (torch.ones(hidden_size,dtype=torch.float32,device='cuda')*0.3)

output_update_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*0.1)
output_upward_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*(-0.2))
output_upward_bias = (torch.ones(output_channels,dtype=torch.float32,device='cuda')*0.3)

hidden_update_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*1)
hidden_upward_weights = (torch.ones(hidden_size,output_channels,dtype=torch.float32,device='cuda')*2)
hidden_upward_bias = (torch.ones(output_channels,dtype=torch.float32,device='cuda')*3)

torch.ops.load_library('./libfuseSage.so')

print(torch.ops.fuseops.fuseSage(batch,edge_csr_indptr,edge_csr_indices,GPU_node_feature,CPU_node_feature,idx2id_d,idx2id,num_layers,sample_size,sample_size_d,input_update_weights,input_upward_weights,input_upward_bias,output_update_weights,output_upward_weights,output_upward_bias,hidden_update_weights,hidden_upward_weights,hidden_upward_bias))