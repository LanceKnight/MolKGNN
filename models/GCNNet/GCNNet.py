import torch
from torch.nn import ModuleList, ParameterList
from torch.optim import Adam
from torch_geometric.nn import GCNConv, global_mean_pool
import pytorch_lightning as pl
from lr import PolynomialDecayLR
# import pytorch_warmup as warmup
from torch_geometric.utils import add_remaining_self_loops, degree, add_self_loops
from torch_scatter import scatter
from torch.nn import Parameter
import torch.nn.functional as F
import math


def propagate(x, edge_index, edge_weight=None):
    """ feature propagation procedure: sparsematrix
    """
    # print(edge_index.shape)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(row, x.size(0), dtype=x.dtype)

    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    if(edge_weight == None):
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[col]

    return scatter(out, edge_index[0], dim=0, dim_size=x.size(0), reduce='add')


class GCNNet(torch.nn.Module):
    """
    A dummy GCNNet used for testing the general training framework

    It consist of num_layers GCNConv layers and a mean pooling layer
    It outputs a graph embedding
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNNet, self).__init__()
        self.lins = ParameterList()
        self.biases = ParameterList()

        self.lins.append(Parameter(torch.FloatTensor(
            input_dim, hidden_dim)))
        self.biases.append(Parameter(torch.Tensor(hidden_dim)))
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.lins.append(
                    Parameter(torch.FloatTensor(hidden_dim, hidden_dim)))
                self.biases.append(Parameter(torch.Tensor(hidden_dim)))

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.lins)):
            stdv = 1. / math.sqrt(self.lins[i].size(1))
            self.lins[i].data.uniform_(-stdv, stdv)
            self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        batch = data.batch

        for i in range(len(self.lins)):
            h = torch.mm(h, self.lins[i])
            h = propagate(h, edge_index, edge_weight=None) + self.biases[i]
            h = F.relu(h)

        graph_embedding = global_mean_pool(h, batch)

        # print(graph_embedding)

        return graph_embedding

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("GCNNet")
        parser.add_argument('--num_layers', type=int, default=3)
        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)

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
