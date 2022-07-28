from .kernels import PredefinedKernelSetConv, KernelSetConv

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch.nn import ModuleList
from torch_geometric.utils import add_self_loops
import torch

class MolGCN(MessagePassing):
    def __init__(self, num_layers=5, num_kernel1_1hop=0, num_kernel2_1hop=0,
                 num_kernel3_1hop=0, num_kernel4_1hop=0, num_kernel1_Nhop=0,
                 num_kernel2_Nhop=0, num_kernel3_Nhop=0, num_kernel4_Nhop=0,
                 predefined_kernelsets=True, x_dim=5, p_dim=3,
                 edge_attr_dim=1, ):
        super(MolGCN, self).__init__(aggr='add')
        self.num_layers = num_layers
        if num_layers < 1:
            raise Exception('at least one convolution layer is needed')

        self.layers = ModuleList()

        self.num_kernels_list = []
        # first layer
        if (num_kernel1_1hop is not None) and (
                num_kernel2_1hop is not None) and (
                num_kernel3_1hop is not None) and (
                num_kernel4_1hop is not None) and (
                predefined_kernelsets == False):
            kernel_layer = KernelSetConv(num_kernel1_1hop, num_kernel2_1hop,
                                         num_kernel3_1hop, num_kernel4_1hop,
                                         D=p_dim, node_attr_dim=x_dim,
                                         edge_attr_dim=edge_attr_dim)
            num_kernels = num_kernel1_1hop + num_kernel2_1hop + \
                          num_kernel3_1hop + num_kernel4_1hop
        elif (predefined_kernelsets == True):
            kernel_layer = PredefinedKernelSetConv(D=p_dim,
                                                   node_attr_dim=x_dim,
                                                   edge_attr_dim=edge_attr_dim,
                                                   L1=num_kernel1_1hop,
                                                   L2=num_kernel2_1hop,
                                                   L3=num_kernel3_1hop,
                                                   L4=num_kernel4_1hop,
                                                   is_first_layer=True)
            num_kernels = kernel_layer.get_num_kernel()
        else:
            raise Exception('MolGCN: num_kernel1-4 need to be specified')
        self.layers.append(kernel_layer)
        self.num_kernels_list.append(
            num_kernels)  # num of kernels in each layer

        # N layer
        x_dim = num_kernels
        for i in range(num_layers - 1):
            # print(f'layer:{i}')
            if (predefined_kernelsets == True):
                # print(f'num_kernels:{self.num_kernels(i)}')
                kernel_layer = PredefinedKernelSetConv(D=p_dim,
                                                       node_attr_dim=self.num_kernels(
                                                           i)+x_dim,
                                                       edge_attr_dim=edge_attr_dim,
                                                       L1=num_kernel1_Nhop,
                                                       L2=num_kernel2_Nhop,
                                                       L3=num_kernel3_Nhop,
                                                       L4=num_kernel4_Nhop,
                                                       is_first_layer=False)
            else:
                kernel_layer = KernelSetConv(L1=num_kernel1_Nhop,
                                             L2=num_kernel2_Nhop,
                                             L3=num_kernel3_Nhop,
                                             L4=num_kernel4_Nhop,
                                             D=p_dim,
                                             node_attr_dim=self.num_kernels(i),
                                             edge_attr_dim=edge_attr_dim)
            self.layers.append(kernel_layer)
            self.num_kernels_list.append(kernel_layer.get_num_kernel())

    def num_kernels(self, layer):
        return self.num_kernels_list[layer]

    def forward(self, *argv, **kwargv):
        if len(argv) != 0:
            raise Exception(
                'Kernel does not take positional argument, use keyword '
                'argument instead. e.g. model(data=data)')

        if len(kwargv) == 2:
            x = kwargv['data'].x
            edge_index = kwargv['data'].edge_index
            edge_attr = kwargv['data'].edge_attr
            p = kwargv['data'].p
            p_focal_deg1 = kwargv['data'].p_focal_deg1
            nei_p_deg1 = kwargv['data'].nei_p_deg1
            nei_edge_attr_deg1 = kwargv['data'].nei_edge_attr_deg1
            selected_index_deg1 = kwargv['data'].selected_index_deg1
            nei_index_deg1 = kwargv['data'].nei_index_deg1

            p_focal_deg2 = kwargv['data'].p_focal_deg2
            nei_p_deg2 = kwargv['data'].nei_p_deg2
            nei_edge_attr_deg2 = kwargv['data'].nei_edge_attr_deg2
            selected_index_deg2 = kwargv['data'].selected_index_deg2
            nei_index_deg2 = kwargv['data'].nei_index_deg2

            p_focal_deg3 = kwargv['data'].p_focal_deg3
            nei_p_deg3 = kwargv['data'].nei_p_deg3
            nei_edge_attr_deg3 = kwargv['data'].nei_edge_attr_deg3
            selected_index_deg3 = kwargv['data'].selected_index_deg3
            nei_index_deg3 = kwargv['data'].nei_index_deg3

            p_focal_deg4 = kwargv['data'].p_focal_deg4
            nei_p_deg4 = kwargv['data'].nei_p_deg4
            nei_edge_attr_deg4 = kwargv['data'].nei_edge_attr_deg4
            selected_index_deg4 = kwargv['data'].selected_index_deg4
            nei_index_deg4 = kwargv['data'].nei_index_deg4

            save_score = kwargv['save_score']
        else:
            x = kwargv['x']
            edge_index = kwargv['edge_index']
            edge_attr = kwargv['edge_attr']
            p = kwargv['p']

            p_focal_deg1 = kwargv['p_focal_deg1']
            nei_p_deg1 = kwargv['nei_p_deg1']
            nei_edge_attr_deg1 = kwargv['nei_edge_attr_deg1']
            selected_index_deg1 = kwargv['selected_index_deg1']
            nei_index_deg1 = kwargv['nei_index_deg1']

            p_focal_deg2 = kwargv['p_focal_deg2']
            nei_p_deg2 = kwargv['nei_p_deg2']
            nei_edge_attr_deg2 = kwargv['nei_edge_attr_deg2']
            selected_index_deg2 = kwargv['selected_index_deg2']
            nei_index_deg2 = kwargv['nei_index_deg2']

            p_focal_deg3 = kwargv['p_focal_deg3']
            nei_p_deg3 = kwargv['nei_p_deg3']
            nei_edge_attr_deg3 = kwargv['nei_edge_attr_deg3']
            selected_index_deg3 = kwargv['selected_index_deg3']
            nei_index_deg3 = kwargv['nei_index_deg3']

            p_focal_deg4 = kwargv['p_focal_deg4']
            nei_p_deg4 = kwargv['nei_p_deg4']
            nei_edge_attr_deg4 = kwargv['nei_edge_attr_deg4']
            selected_index_deg4 = kwargv['selected_index_deg4']
            nei_index_deg4 = kwargv['nei_index_deg4']

            data = Data(x=x, p=p, edge_index=edge_index, edge_attr=edge_attr,
                        p_focal_deg1=p_focal_deg1, p_focal_deg2=p_focal_deg2,
                        p_focal_deg3=p_focal_deg3, p_focal_deg4=p_focal_deg4,
                        nei_p_deg1=nei_p_deg1, nei_p_deg2=nei_p_deg2,
                        nei_p_deg3=nei_p_deg3, nei_p_deg4=nei_p_deg4,
                        nei_edge_attr_deg1=nei_edge_attr_deg1,
                        nei_edge_attr_deg2=nei_edge_attr_deg2,
                        nei_edge_attr_deg3=nei_edge_attr_deg3,
                        nei_edge_attr_deg4=nei_edge_attr_deg4,
                        selected_index_deg1=selected_index_deg1,
                        selected_index_deg2=selected_index_deg2,
                        selected_index_deg3=selected_index_deg3,
                        selected_index_deg4=selected_index_deg4,
                        nei_index_deg1=nei_index_deg1,
                        nei_index_deg2=nei_index_deg2,
                        nei_index_deg3=nei_index_deg3,
                        nei_index_deg4=nei_index_deg4
                        )
            # print(f'foward: data.x{data.x}')
            save_score = kwargv['save_score']
        h = x
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        for i in range(self.num_layers):
            # print(f'KernelLayer.py::{i}th layer==============')
            data.x = h

            kernel_layer = self.layers[i]
            if i == self.num_layers-1:
                is_last_layer = True
            else:
                is_last_layer = False
            sim_sc = kernel_layer(is_last_layer = is_last_layer, data=data, save_score=save_score)

            # print(f'edge_index:{edge_index.device}, sim_sc:{sim_sc.device}')
            # print('sim_sc')
            # print(sim_sc)
            h = torch.cat(h, self.propagate(edge_index=edge_index, sim_sc=sim_sc))
            # print(f'layer time:{end-start}')
        return h



    def get_kernels(self, layer_id, degree, file_name):
        # TODO: not working yet 
        kernel_set_conv = self.layers[layer_id]
        x_center1 = kernel_set_conv.kernelconv1.x_center
        x_support = kernel_set_conv.x_support
        p_support = kernel_set_conv.p_support
        kernels = {}



    def message(self, sim_sc_j):
        # print(f'sim_sc_j:{sim_sc_j.shape}')
        return sim_sc_j
