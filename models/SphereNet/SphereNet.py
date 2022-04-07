
from ..ChIRoNet.train_functions import get_local_structure_map
from ..ChIRoNet.gnn_3D.spherenet import SphereNet as Encoder

import torch
from torch_geometric.nn.acts import swish

class SphereNet(torch.nn.Module):
    r"""
        codes adapted from https://github.com/keiradams/ChIRo
         The spherical message passing neural network SphereNet from the
         `"Spherical Message Passing for 3D Graph Networks"
         <https://arxiv.org/abs/2102.05013>`_ paper.

        Args:
            energy_and_force (bool, optional): If set to :obj:`True`,
            will predict energy and take the negative of the derivative
            of the energy with respect to the atomic positions as
            predicted forces. (default: :obj:`False`)
            cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
            num_layers (int, optional): Number of building blocks. (
            default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (
            default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (
            default: :obj:`1`)
            int_emb_size (int, optional): Embedding size used for
            interaction triplets. (default: :obj:`64`)
            basis_emb_size_dist (int, optional): Embedding size used in
            the basis transformation of distance. (default: :obj:`8`)
            basis_emb_size_angle (int, optional): Embedding size used in
            the basis transformation of angle. (default: :obj:`8`)
            basis_emb_size_torsion (int, optional): Embedding size used
            in the basis transformation of torsion. (default: :obj:`8`)
            out_emb_channels (int, optional): Embedding size used for
            atoms in the output block. (default: :obj:`256`)
            num_spherical (int, optional): Number of spherical
            harmonics. (default: :obj:`7`)
            num_radial (int, optional): Number of radial basis
            functions. (default: :obj:`6`)
            envelop_exponent (int, optional): Shape of the smooth
            cutoff. (default: :obj:`5`)
            num_before_skip (int, optional): Number of residual layers
            in the interaction blocks before the skip connection. (
            default: :obj:`1`)
            num_after_skip (int, optional): Number of residual layers in
            the interaction blocks before the skip connection. (default:
            :obj:`2`)
            num_output_layers (int, optional): Number of linear layers
            for the output blocks. (default: :obj:`3`)
            act: (function, optional): The activation funtion. (default:
            :obj:`swish`)
            output_init: (str, optional): The initialization fot the
            output. It could be :obj:`GlorotOrthogonal` and
            :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)
        """

    def __init__(
            self, energy_and_force=False, cutoff=5.0, num_layers=4,
            hidden_channels=128, out_channels=1, int_emb_size=64,
            basis_emb_size_dist=8, basis_emb_size_angle=8,
            basis_emb_size_torsion=8, out_emb_channels=256,
            num_spherical=7, num_radial=6, envelope_exponent=5,
            num_before_skip=1, num_after_skip=2, num_output_layers=3,
            act=swish, output_init='GlorotOrthogonal', use_node_features=True,
            MLP_hidden_sizes=[]):
        super(SphereNet, self).__init__()
        self.encoder = Encoder(
                                energy_and_force=energy_and_force,  # False
                                cutoff=cutoff,  # 5.0
                                num_layers=num_layers,  # 4
                                hidden_channels=hidden_channels,  # 128
                                out_channels=out_channels,  # 1
                                int_emb_size=int_emb_size,  # 64
                                basis_emb_size_dist=basis_emb_size_dist,  # 8
                                basis_emb_size_angle=basis_emb_size_angle,  # 8
                                basis_emb_size_torsion=basis_emb_size_torsion,  # 8
                                out_emb_channels=out_emb_channels,  # 256
                                num_spherical=num_spherical,  # 7
                                num_radial=num_radial,  # 6
                                envelope_exponent=envelope_exponent,  # 5
                                num_before_skip=num_before_skip,  # 1
                                num_after_skip=num_after_skip,  # 2
                                num_output_layers=num_output_layers,  # 3
                                act=swish,
                                output_init=output_init,
                                use_node_features=use_node_features,
                                MLP_hidden_sizes=MLP_hidden_sizes,  # [] for contrastive
                            )
    def forward(self, batch_data):
        psi_indices = batch_data.dihedral_angle_index
        LS_map, alpha_indices = get_local_structure_map(psi_indices)

        device = batch_data.x.device
        LS_map = LS_map.to(device)
        alpha_indices = alpha_indices.to(device)

        print(f'SphereNet.py::data:{batch_data}')
        print(f'SphereNet.py::LS_map:{LS_map.shape}')
        print(f'SphereNet.py::alpha_indices:{alpha_indices.shape}')

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
        parser = parent_parser.add_argument_group("SphereNet")

        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)
        parser.add_argument('--energy_and_force', action="store_true",
                            help='')
        parser.add_argument('--cutoff', type=float, default=5.0,
                            help='')
        parser.add_argument('--num_layers', type=int, default=128,
                            help='')
        parser.add_argument('--hidden_channels', type=int, default=4,
                            help='')
        parser.add_argument('--out_channels', type=int, default=1,
                            help='')
        parser.add_argument('--int_emb_size', type=int, default=64,
                            help='')
        parser.add_argument('--basis_emb_size_dist', type=int, default=8,
                            help='')
        parser.add_argument('--basis_emb_size_angle', type=int, default=8,
                            help='')
        parser.add_argument('--basis_emb_size_torsion', type=int, default=8,
                            help='')
        parser.add_argument('--out_emb_channels', type=int, default=256,
                            help='')
        parser.add_argument('--num_spherical', type=int, default=7,
                            help='')
        parser.add_argument('--num_radial', type=int, default=6,
                            help='')
        parser.add_argument('--envelope_exponent', type=int, default=5,
                            help='')
        parser.add_argument('--num_before_skip', type=int, default=1,
                            help='')
        parser.add_argument('--num_after_skip', type=int, default=2,
                            help='')
        parser.add_argument('--num_output_layers', type=int, default=3,
                            help='')
        parser.add_argument('--MLP_hidden_sizes', type=list, default=[32,32],
                            help='')

        return parent_parser