from .schnet import SchNet as Encoder
import torch





class SchNet(torch.nn.Module):

    def __init__(
            self, energy_and_force=False, cutoff=10.0, num_layers=6, hidden_channels=128, num_filters=128,
            num_gaussians=50, out_channels=32):
        super(SchNet, self).__init__()
        self.encoder = Encoder(energy_and_force=energy_and_force,
                               cutoff=cutoff,
                               num_layers=num_layers,
                               hidden_channels=hidden_channels,
                               num_filters=num_filters,
                               num_gaussians=num_gaussians,
                               out_channels=out_channels
                            )
    def forward(self, batch_data):



        batch_data.z = batch_data.x.squeeze()
        graph_embedding = self.encoder(batch_data)

        return graph_embedding

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model arguments to the parent parser
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("SchNet")

        # Add specific model arguments below
        # E.g., parser.add_argument('--GCN_arguments', type=int,
        # default=12)
        parser.add_argument('--energy_and_force', action="store_true",
                            help='')
        parser.add_argument('--cutoff', type=float, default=10.0,
                            help='')
        parser.add_argument('--num_layers', type=int, default=6,
                            help='')
        parser.add_argument('--hidden_channels', type=int, default=128,
                            help='')
        parser.add_argument('--num_filters', type=int, default=128,
                            help='')
        parser.add_argument('--num_gaussians', type=int, default=50,
                            help='')
        parser.add_argument('--out_channels', type=int, default=32,
                            help='')


        return parent_parser