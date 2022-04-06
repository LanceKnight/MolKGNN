from ..ChIRoNet.gnn_3D.dimenet_pp import DimeNetPlusPlus

import json

from .alpha_encoder import Encoder
from .train_functions import get_local_structure_map

import torch
from torch.nn import ModuleList
from torch.optim import Adam
from torch_geometric.nn import ChebConv, global_mean_pool
import pytorch_lightning as pl
# from lr import PolynomialDecayLR
# import pytorch_warmup as warmup


class DimeNetPP(torch.nn.Module):
    """
        codes adapted from https://github.com/PattanaikL/chiral_gnn
        DimeNet++ implementation based on https://github.com/klicperajo/dimenet.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """
    def __init__(self,
            hidden_channels,
            out_channels,
            num_blocks,
            int_emb_size,
            basis_emb_size,
            out_emb_channels,
            num_spherical,
            num_radial,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            act=swish,
            MLP_hidden_sizes = [],):
        super(DimeNetPP, self).__init__()
        self.encoder = DimeNetPlusPlus(
            hidden_channels=params['hidden_channels'],  # 128
            out_channels=params['out_channels'],  # 1
            num_blocks=params['num_blocks'],  # 4
            int_emb_size=params['int_emb_size'],  # 64
            basis_emb_size=params['basis_emb_size'],  # 8
            out_emb_channels=params['out_emb_channels'],  # 256
            num_spherical=params['num_spherical'],  # 7
            num_radial=params['num_radial'],  # 6
            cutoff=params['cutoff'],  # 5.0
            envelope_exponent=params['envelope_exponent'],  # 5
            num_before_skip=params['num_before_skip'],  # 1
            num_after_skip=params['num_after_skip'],  # 2
            num_output_layers=params['num_output_layers'],  # 3
            act=swish,
            MLP_hidden_sizes=[],  # [] for contrastive
        )


    def forward(self, batch_data):
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        device = batch_data.x.device
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)

        output, latent_vector, phase_shift_norm, z_alpha, mol_embedding, \
        c_tensor, phase_cos, phase_sin, sin_cos_psi, sin_cos_alpha = self.encoder(batch_data, LS_map, alpha_indices)
        # graph_embedding = self.encoder(batch_data, LS_map, alpha_indices)
        graph_embedding = mol_embedding

        return graph_embedding


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("DimeNetPP")

        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)
        parser.add_argument('--num_blocks', type=int, default=4,
                            help='')
        # parser.add_argument('--F_H', type=int, default=32,
        #                     help='dimension of final node embeddings, after EConv and GAT layers')
        # parser.add_argument('--F_H_embed', type=int, default=52,
        #                     help='dimension of initial node feature vector, '
        #                          'currently 52')
        # parser.add_argument('--F_E_embed', type=int, default=14,
        #                     help='dimension of initial edge feature vector, '
        #                          'currently 14')
        # parser.add_argument('--F_H_EConv', type=int, default=64,
        #                     help='dimension of node embedding after EConv layer')
        parser.add_argument('--layers_dict', type=json.loads, default='{'
                                                                   '"EConv_mlp_hidden_sizes": [32, 32],'
                                                                   '"GAT_hidden_node_sizes": [8],'
                                                                   '"encoder_hidden_sizes_D": [8, 8],'
                                                                   '"encoder_hidden_sizes_phi": [8, 8],'
                                                                   '"encoder_hidden_sizes_c": [8, 8],'
                                                                   '"encoder_hidden_sizes_alpha": [8, 8],'
                                                                   '"encoder_hidden_sizes_sinusoidal_shift": [8, 8],'
                                                                   '"output_mlp_hidden_sizes": [8, 8]'
                                                                   '}',
                            help='')
        parser.add_argument('--activation_dict', type=json.loads, default='{'
                                                                   '"encoder_hidden_activation_D": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"encoder_hidden_activation_phi": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"encoder_hidden_activation_c": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"encoder_hidden_activation_alpha": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"encoder_hidden_activation_sinusoidal_shift": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"encoder_output_activation_D": "torch.nn.Identity()",'
                                                                   '"encoder_output_activation_phi": "torch.nn.Identity()",'
                                                                   '"encoder_output_activation_c": "torch.nn.Identity()",'  
                                                                   '"encoder_output_activation_alpha": "torch.nn.Identity()",' 
                                                                   '"encoder_output_activation_sinusoidal_shift": "torch.nn.Identity()",' 
                                                                   '"EConv_mlp_hidden_activation": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"EConv_mlp_output_activation": "torch.nn.Identity()",'
                                                                   '"output_mlp_hidden_activation": "torch.nn.LeakyReLU(negative_slope=0.01)",'
                                                                   '"output_mlp_output_activation": "torch.nn.Identity()"'
                                                                   '}',
                            help='')
        parser.add_argument('--GAT_N_heads', type=int, default=4,
                            help='')
        parser.add_argument('--use_chiral_message_passing', action='store_true')
        parser.add_argument('--CMP_EConv_MLP_hidden_sizes', type=str,
                            default="[8,8]", help='')
        parser.add_argument('--CMP_GAT_N_layers', type=int, default=3,
                            help='')
        parser.add_argument('--CMP_GAT_N_heads', type=int, default=2,
                            help='')
        parser.add_argument('--c_coefficient_normalization', type=str,
                            default="sigmoid", help='# None, or one of ["sigmoid", "softmax"]')
        parser.add_argument('--encoder_reduction', type=str, default='sum',
                            help='mean or sum')
        parser.add_argument('--output_concatenation_mode', type=str,
                            default='molecule', help='')
        parser.add_argument('--EConv_bias', action='store_true',
                            help='')
        parser.add_argument('--GAT_bias', action='store_true',
                            help='')
        parser.add_argument('--encoder_biases', action='store_true',
                            help='')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='applied to hidden layers (not input/output layer) of Encoder MLPs, hidden layers (not input/output layer) of EConv MLP, and all GAT layers (using their dropout parameter)')
        parser.add_argument('--num_layers', type=int, default=3,
                            help='not used')

        return parent_parser


    # def configure_optimizers(self, warmup_iterations, tot_iterations,
    #                          peak_lr, end_lr):
    #     """
    #     Returns an optimizer and scheduler suitable for GCNNet
    #     :return: optimizer, scheduler
    #     """
    #     optimizer = Adam(self.parameters())
    #     # scheduler = warmup.
    #     scheduler = {
    #         'scheduler': PolynomialDecayLR(
    #             optimizer,
    #             warmup_iterations=warmup_iterations,
    #             tot_iterations=tot_iterations,
    #             lr=peak_lr,
    #             end_lr=end_lr,
    #             power=1.0,
    #         ),
    #         'name': 'learning_rate',
    #         'interval': 'step',
    #         'frequency': 1,
    #     }
    #     return optimizer, scheduler


