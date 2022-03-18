import torch
from torch.nn import ModuleList
from torch.optim import Adam
from torch_geometric.nn import ChebConv, global_mean_pool
import pytorch_lightning as pl
from lr import PolynomialDecayLR
# import pytorch_warmup as warmup


class ChebNet(torch.nn.Module):
    """
    A dummy GCNNet used for testing the general training framework

    It consist of num_layers GCNConv layers and a mean pooling layer
    It outputs a graph embedding
    """
    def __init__(self, input_dim, hidden_dim, num_layers, K):
        super(ChebNet, self).__init__()
        self.layers = ModuleList()
        gcn_conv = ChebConv(input_dim, hidden_dim, K)
        self.layers.append(gcn_conv)
        if num_layers>1:
            for i in range(num_layers-1):
                gcn_conv = ChebConv(hidden_dim, hidden_dim)
                self.layers.append(gcn_conv)


    def forward(self, data):
        h = data.x
        edge_index = data.edge_index
        batch = data.batch

        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index)

        graph_embedding = global_mean_pool(h, batch)

        return graph_embedding


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("ChebNet")
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--K', type=int, default=10)
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


