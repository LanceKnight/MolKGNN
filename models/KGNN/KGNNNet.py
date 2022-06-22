from .KernelLayer import MolGCN
from lr import PolynomialDecayLR

import torch
from torch.nn import Linear, Sigmoid, BatchNorm1d
from torch_geometric.nn import global_add_pool
from torch.optim import Adam


class KGNNNet(torch.nn.Module):
    def __init__(self, num_layers=1, num_kernel1_1hop=0, num_kernel2_1hop=0,
                 num_kernel3_1hop=0, num_kernel4_1hop=0, num_kernel1_Nhop=0,
                 num_kernel2_Nhop=0, num_kernel3_Nhop=0, num_kernel4_Nhop=0,
                 predefined_kernelsets=True, x_dim=5, p_dim=3, edge_attr_dim=1,
                 drop_ratio=0, graph_embedding_dim=5):
        super(KGNNNet, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.D = p_dim
        self.graph_embedding_linear = Linear(
            num_kernel1_Nhop
            + num_kernel2_Nhop
            + num_kernel3_Nhop
            + num_kernel4_Nhop
            , graph_embedding_dim)
        self.node_batch_norm = BatchNorm1d(x_dim)
        self.edge_batch_norm = BatchNorm1d(edge_attr_dim)

        if self.num_layers < 1:
            raise ValueError(
                "GNN_graphpred: Number of GNN layers must be greater than 0.")

        self.gnn = MolGCN(num_layers=num_layers,
                          num_kernel1_1hop=num_kernel1_1hop,
                          num_kernel2_1hop=num_kernel2_1hop,
                          num_kernel3_1hop=num_kernel3_1hop,
                          num_kernel4_1hop=num_kernel4_1hop,
                          num_kernel1_Nhop=num_kernel1_Nhop,
                          num_kernel2_Nhop=num_kernel2_Nhop,
                          num_kernel3_Nhop=num_kernel3_Nhop,
                          num_kernel4_Nhop=num_kernel4_Nhop, x_dim=x_dim,
                          p_dim=p_dim, edge_attr_dim=edge_attr_dim,
                          predefined_kernelsets=predefined_kernelsets)

        self.pool = global_add_pool
        self.atom_encoder = Linear(x_dim, graph_embedding_dim)
        self.bond_encoder = Linear(edge_attr_dim, graph_embedding_dim)

    def save_kernellayer(self, path, time_stamp):
        layers = self.gnn.layers
        print(f'{self.D}D, there are {len(layers)} layers')
        for i, layer in enumerate(layers):
            print(f'saving {i}th layer')
            torch.save(layer.state_dict(),
                       f'{path}/{time_stamp}_{i}th_layer.pth')

    def forward(self, *argv, save_score=False):
        if len(argv) == 33:
            x, p, edge_index, edge_attr, batch, \
            x_focal_deg1, x_focal_deg2, x_focal_deg3, x_focal_deg4, \
            p_focal_deg1, p_focal_deg2, p_focal_deg3, p_focal_deg4, \
            nei_x_deg1, nei_x_deg2, nei_x_deg3, nei_x_deg4, \
            nei_p_deg1, nei_p_deg2, nei_p_deg3, nei_p_deg4, \
            nei_edge_attr_deg1, nei_edge_attr_deg2, nei_edge_attr_deg3, \
            nei_edge_attr_deg4, \
            selected_index_deg1, selected_index_deg2, selected_index_deg3, \
            selected_index_deg4, \
            nei_index_deg1, nei_index_deg2, nei_index_deg3, nei_index_deg4 \
                = \
                argv[0], argv[1], argv[2], argv[3], argv[4], \
                argv[5], argv[6], argv[7], argv[8], \
                argv[9], argv[10], argv[11], argv[12], \
                argv[13], argv[14], argv[15], argv[16], \
                argv[17], argv[18], argv[19], argv[20], \
                argv[21], argv[22], argv[23], argv[24], \
                argv[25], argv[26], argv[27], argv[28], \
                argv[29], argv[30], argv[31], argv[32],
        elif len(argv) == 1:
            data = argv[0]
            x, p, edge_index, edge_attr, batch, \
            p_focal_deg1, p_focal_deg2, p_focal_deg3, p_focal_deg4, \
            nei_p_deg1, nei_p_deg2, nei_p_deg3, nei_p_deg4, \
            nei_edge_attr_deg1, nei_edge_attr_deg2, nei_edge_attr_deg3, \
            nei_edge_attr_deg4, \
            selected_index_deg1, selected_index_deg2, selected_index_deg3, \
            selected_index_deg4, \
            nei_index_deg1, nei_index_deg2, nei_index_deg3, nei_index_deg4 \
                = \
                data.x, data.p, data.edge_index, data.edge_attr, data.batch, \
                data.p_focal_deg1, data.p_focal_deg2, data.p_focal_deg3, \
                data.p_focal_deg4, \
                data.nei_p_deg1, data.nei_p_deg2, data.nei_p_deg3, \
                data.nei_p_deg4, \
                data.nei_edge_attr_deg1, data.nei_edge_attr_deg2, \
                data.nei_edge_attr_deg3, data.nei_edge_attr_deg4, \
                data.selected_index_deg1, data.selected_index_deg2, \
                data.selected_index_deg3, data.selected_index_deg4, \
                data.nei_index_deg1, data.nei_index_deg2, \
                data.nei_index_deg3, data.nei_index_deg4
        else:
            raise ValueError("unmatched number of arguments.")

        # print(f'x:{x.shape}')
        # print(f'self.atom_encoder{self.atom_encoder}')
        # x = self.atom_encoder(data.x)
        # edge_attr = self.bond_encoder(data.edge_attr)
        x = self.node_batch_norm(x)
        edge_attr = self.edge_batch_norm(edge_attr)
        node_representation = self.gnn(x=x, edge_index=edge_index,
                                       edge_attr=edge_attr, p=p,
                                       p_focal_deg1=p_focal_deg1,
                                       p_focal_deg2=p_focal_deg2,
                                       p_focal_deg3=p_focal_deg3,
                                       p_focal_deg4=p_focal_deg4,
                                       nei_p_deg1=nei_p_deg1,
                                       nei_p_deg2=nei_p_deg2,
                                       nei_p_deg3=nei_p_deg3,
                                       nei_p_deg4=nei_p_deg4,
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
                                       nei_index_deg4=nei_index_deg4,
                                       save_score=save_score)

        graph_representation = self.graph_embedding_linear(
            self.pool(node_representation, batch))

        return graph_representation

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("KGNNNet")
        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--num_kernel1_1hop', type=int, default=10)
        parser.add_argument('--num_kernel2_1hop', type=int, default=20)
        parser.add_argument('--num_kernel3_1hop', type=int, default=30)
        parser.add_argument('--num_kernel4_1hop', type=int, default=40)
        parser.add_argument('--num_kernel1_Nhop', type=int, default=10)
        parser.add_argument('--num_kernel2_Nhop', type=int, default=20)
        parser.add_argument('--num_kernel3_Nhop', type=int, default=30)
        parser.add_argument('--num_kernel4_Nhop', type=int, default=40)
        parser.add_argument('--node_feature_dim', type=int, default=27)
        parser.add_argument('--edge_feature_dim', type=int, default=7)
        parser.add_argument('--hidden_dim', type=int, default=64)

        return parent_parser

    def configure_optimizers(self, warmup_iterations, tot_iterations,
                             peak_lr, end_lr):
        """
        Returns an optimizer and scheduler suitable for GCNNet
        :return: optimizer, scheduler
        """
        optimizer = Adam(self.parameters())
        # scheduler = warmup.
        scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_iterations=warmup_iterations,
                tot_iterations=tot_iterations,
                lr=peak_lr,
                end_lr=end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return optimizer, scheduler
