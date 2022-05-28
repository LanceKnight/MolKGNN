from models.KGNN.KGNNNet import KGNNNet
import torch

data = torch.load('debug/debug.data').to('cpu')
print(data)
gnn_model = KGNNNet(num_layers=3,
                                     num_kernel1_1hop = 10,
                                     num_kernel2_1hop = 20,
                                     num_kernel3_1hop = 30,
                                     num_kernel4_1hop = 50,
                                     num_kernel1_Nhop = 10,
                                     num_kernel2_Nhop = 20,
                                     num_kernel3_Nhop = 30,
                                     num_kernel4_Nhop = 50,
                                     x_dim = 27,
                                     edge_attr_dim=7,
                                     graph_embedding_dim = 32,
                                     predefined_kernelsets=False).to('cpu')

output = gnn_model(data)
print(output)

if __name__ == '__main__':
	print('here')