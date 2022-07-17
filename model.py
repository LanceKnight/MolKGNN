from data import get_dataset
from models.GCNNet.GCNNet import GCNNet
from models.KGNN.KGNNNet import KGNNNet
from models.DimeNetPP.DimeNetPP import DimeNetPP
from models.ChebNet.ChebNet import ChebNet
from models.ChIRoNet.ChIRoNet import ChIRoNet
from models.ChIRoNet.params_interpreter import string_to_object
from models.SphereNet.SphereNet import SphereNet
from evaluation import calculate_logAUC, calculate_ppv, calculate_accuracy, \
    calculate_f1_score, calculate_auc
from lr import PolynomialDecayLR

# Public libraries
from copy import deepcopy
import os
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error
from torch.nn import Linear, Sigmoid, ReLU, Embedding, Dropout
from torch_geometric.data import Data
import torch
from torch.optim import Adam
from torch_geometric.nn.acts import swish
import time

class GNNModel(pl.LightningModule):
    """
    A wrapper for different GNN models

    It uses a GNN model to output a graph embedding, and use some prediction
    method to output a final prediction

    Here a linear layer with a sigmoid function is used
    """

    def __init__(self,
                 gnn_type,
                 args=None
                 ):
        super(GNNModel, self).__init__()
        if gnn_type == 'gcn':
            self.gnn_model = GCNNet(args.node_feature_dim, args.hidden_dim,
                                    args.num_layers)
        elif gnn_type == 'chebnet':
            self.gnn_model = ChebNet(args.node_feature_dim, args.hidden_dim, args.num_layers, args.K)
        # elif gnn_type == 'dimenet':
        #     self.gnn_model = DimeNet(emb_size=args.hidden_dim,
        #                              num_blocks=args.num_layers,
        #                              num_bilinear=1, num_spherical=7,
        #     num_radial=6, cutoff=5.0, envelope_exponent=5, num_before_skip=1,
        #     num_after_skip=2, num_dense_output=3, num_targets=12,
        #     output_init='zeros', name='dimenet')
        elif gnn_type == 'chironet':

            layers_dict = deepcopy(args.layers_dict)

            activation_dict = deepcopy(args.activation_dict)

            for key, value in args.activation_dict.items():
                activation_dict[key] = string_to_object[
                    value]  # convert strings to actual python
                # objects/functions using pre-defined mapping
            print(
                f'model.py::chironet argument:'
                f'{type(args.activation_dict["EConv_mlp_hidden_activation"])}')
            self.gnn_model = ChIRoNet(
                F_z_list=args.F_z_list,  # dimension of latent space
                F_H=args.F_H,
                # dimension of final node embeddings, after EConv and GAT layers
                F_H_embed=args.F_H_embed,
                # dimension of initial node feature vector, currently 41
                F_E_embed=args.F_E_embed,
                # dimension of initial edge feature vector, currently 12
                F_H_EConv=args.F_H_EConv,
                # dimension of node embedding after EConv layer
                layers_dict=args.layers_dict,
                activation_dict= activation_dict,
                GAT_N_heads=args.GAT_N_heads,
                chiral_message_passing=args.use_chiral_message_passing,
                CMP_EConv_MLP_hidden_sizes=args.CMP_EConv_MLP_hidden_sizes,
                CMP_GAT_N_layers=args.CMP_GAT_N_layers,
                CMP_GAT_N_heads=args.CMP_GAT_N_heads,
                c_coefficient_normalization=args.c_coefficient_normalization,
                encoder_reduction=args.encoder_reduction,
                output_concatenation_mode=args.output_concatenation_mode,
                EConv_bias=args.EConv_bias,
                GAT_bias=args.GAT_bias,
                encoder_biases=args.encoder_biases,
                dropout=args.dropout,
            )
            out_dim = args.F_H
        elif gnn_type == 'dimenet_pp':
            print(f'model.py::running dimenet_pp')
            self.gnn_model = DimeNetPP(
                hidden_channels=args.hidden_channels,
                out_channels=args.out_channels,
                num_blocks=args.num_blocks,
                int_emb_size=args.int_emb_size,
                basis_emb_size=args.basis_emb_size,
                out_emb_channels=args.out_emb_channels,
                num_spherical=args.num_spherical,
                num_radial=args.num_radial,
                cutoff=args.cutoff,
                envelope_exponent=args.envelope_exponent,
                num_before_skip=args.num_before_skip,
                num_after_skip=args.num_after_skip,
                num_output_layers=args.num_output_layers,
                # act_name='swish',
                act=swish,
                MLP_hidden_sizes=[],  # [] for contrastive)
            )
            out_dim = args.out_channels
        elif gnn_type == 'spherenet':
            self.gnn_model = SphereNet(
                energy_and_force=False,  # False
                cutoff=args.cutoff,  # 5.0
                num_layers=args.num_layers,  # 4
                hidden_channels=args.hidden_channels,  # 128
                out_channels=args.out_channels,  # 1
                int_emb_size=args.int_emb_size,  # 64
                basis_emb_size_dist=args.basis_emb_size_dist,  # 8
                basis_emb_size_angle=args.basis_emb_size_angle,  # 8
                basis_emb_size_torsion=args.basis_emb_size_torsion,  # 8
                out_emb_channels=args.out_emb_channels,  # 256
                num_spherical=args.num_spherical,  # 7
                num_radial=args.num_radial,  # 6
                envelope_exponent=args.envelope_exponent,  # 5
                num_before_skip=args.num_before_skip,  # 1
                num_after_skip=args.num_after_skip,  # 2
                num_output_layers=args.num_output_layers,  # 3
                act_name='swish',
                output_init='GlorotOrthogonal',
                use_node_features=True,
                MLP_hidden_sizes=args.MLP_hidden_sizes,
                # [] for contrastive

            )
        elif gnn_type == 'kgnn':
            self.gnn_model = KGNNNet(num_layers=args.num_layers,
                                     num_kernel1_1hop = args.num_kernel1_1hop,
                                     num_kernel2_1hop = args.num_kernel2_1hop,
                                     num_kernel3_1hop = args.num_kernel3_1hop,
                                     num_kernel4_1hop = args.num_kernel4_1hop,
                                     num_kernel1_Nhop = args.num_kernel1_Nhop,
                                     num_kernel2_Nhop = args.num_kernel2_Nhop,
                                     num_kernel3_Nhop = args.num_kernel3_Nhop,
                                     num_kernel4_Nhop = args.num_kernel4_Nhop,
                                     x_dim = args.node_feature_dim,
                                     edge_attr_dim=args.edge_feature_dim,
                                     graph_embedding_dim = args.hidden_dim,
                                     predefined_kernelsets=False
            )
            out_dim = args.hidden_dim
        else:
            raise ValueError(f"model.py::GNNModel: GNN model type is not "
                             f"defined. gnn_type={gnn_type}")
        # self.atom_encoder = Embedding(118, hidden_dim)
        self.lin1 = Linear(args.ffn_hidden_dim, args.ffn_hidden_dim)
        self.lin2 = Linear(args.ffn_hidden_dim, args.task_dim)
        self.ffn = Linear(out_dim, args.task_dim)
        self.dropout = Dropout(p= args.ffn_dropout_rate)
        self.activate_func = ReLU()
        self.warmup_iterations = args.warmup_iterations
        self.tot_iterations = args.tot_iterations
        self.peak_lr = args.peak_lr
        self.end_lr = args.end_lr
        self.loss_func = args.loss_func
        self.graph_embedding = None
        self.smiles_list = None
        self.metrics = args.metrics
        self.valid_epoch_outputs = {}
        self.record_valid_pred = args.record_valid_pred
        self.train_metric = args.train_metric

    def forward(self, data):

        graph_embedding = self.gnn_model(data)
        graph_embedding = self.dropout(graph_embedding)
        prediction = self.ffn(graph_embedding)

        # # Debug
        # print(f'model.py::smiles:{data.smiles}\n ')
        # print(f'prediction:\n{prediction}\n ')
        # print(f'graph_embedding:\n:{graph_embedding}')

        self.graph_embedding = graph_embedding
        self.smiles_list = data.smiles
        return prediction, graph_embedding

    def training_step(self, batch_data, batch_idx):
        """
        Training operations for each iteration includes getting the loss and
        metrics.
        The backpropagation is NOT explicitly specified in this function but
        will be taken care of by the pytorch lightning library, as long as
        there is a "loss" key in the output dictionary
        :param batch_data: the data from each mini-batch
        :param batch_idx: the mini-batch id
        :return: a list of dictionaries. Each dicitonary is the output from
        each iteration, and consists of loss, logAUC and ppv
        """

        # Get prediction and ground truth
        # print(batch_data.edge_index)

        # start = time.time()
        pred_y, _ = self(batch_data)
        # end = time.time()
        # print(f'=model.py::training time:{end-start}')
        pred_y = pred_y.view(-1)
        true_y = batch_data.y.view(-1)

        # print(f'pred_y')
        # print(f'{pred_y}')
        # print(f'true_y')
        # print(f'{true_y}')

        # Get metrics
        results = {}
        loss = self.loss_func(pred_y, true_y.float())
        results['loss'] = loss
        # results = self.get_evaluations(results, true_y, pred_y)

        # self.log(f"train performance by step", results, on_step=True, prog_bar=True, logger=True)
        return results

    def training_epoch_end(self, train_step_outputs):
        """
        Get the mean of loss, logAUC, ppv from all iterations in an epoch
        :param train_step_outputs:
        :return: None, But set self.train_epoch_outputs to a dictionary of
        the mean metrics, for monitoring purposes.
        """
        train_epoch_outputs = {}
        for key in train_step_outputs[0].keys():  # Here train_step_outputs
            # is a list of dictionaries, with each dictionary being the
            # output from each iteration. So train_step_outputs[0] is to get
            # the first dictionary. See return function description from
            # function training_step() above
            mean_output = sum(output[key] for output in train_step_outputs) \
                / len(train_step_outputs)
            train_epoch_outputs[key] = mean_output
            self.log(key, mean_output)

        self.train_epoch_outputs = train_epoch_outputs

        # self.log(f"train performance by epoch", train_epoch_outputs, on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch_data, batch_idx, dataloader_idx):
        """
        Process the data in validation dataloader in evaluation mode
        :param batch_data:
        :param batch_idx:
        :param dataloader_idx:
        :return: It returns a list of lists. The 0th item in the list is the
        outputs (a list) from the validation datasets while the 1st item is the
        outputs from the training datasets. Each output is another list,
        which each item being the dictionary from each step.
        """

        # Only run validation dataset if train_metric is not set
        if ((not self.train_metric) and (dataloader_idx == 0)) or (self.train_metric):
            output = self(batch_data)
            pred_y = output[0].view(-1)
            true_y = batch_data.y.view(-1)

            # Get numpy_prediction and numpy_y and concate those from all batches
            valid_step_output = {}
            valid_step_output['pred_y'] = pred_y
            valid_step_output['true_y'] = true_y
            return valid_step_output

    def validation_epoch_end(self, valid_step_outputs):
        # Only run validation dataset if train_metric is not set
        if self.train_metric:
            for i, outputs_each_dataloader in enumerate(valid_step_outputs):
                results = {}
                all_pred = [output['pred_y'] for output in outputs_each_dataloader]
                all_true = [output['true_y'] for output in outputs_each_dataloader]
                results = self.get_evaluations(results, torch.cat(all_true), torch.cat(all_pred))
                if i == 0:
                    self.valid_epoch_outputs = results
                    # Only log validation dataloader b/c this log is used for
                    # monitoring metric and saving the best model. The actual logging happends within
                    # clearml. See Monitor.py
                    for key in results.keys():
                        self.log(key, results[key])

                    # Store prediciton and labels if needed
                    if self.record_valid_pred:
                        filename = f'logs/valid_predictions/epoch_{self.current_epoch}'
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, 'w+') as out_file:
                            for i, pred in enumerate(all_pred):
                                true = all_true[i]
                                out_file.write(f'{pred},{true}\n')
                else:
                    for key in results.keys():
                        new_key = key + "_no_dropout"
                        self.valid_epoch_outputs[new_key] = results[key]
        else: # Only run validation dataset if train_metric is not set
            valid_step_outputs = valid_step_outputs[0]
            results = {}
            all_pred = [output['pred_y'] for output in valid_step_outputs]
            all_true = [output['true_y'] for output in valid_step_outputs]

            # Store prediciton and labels if needed
            if self.record_valid_pred:
                filename = f'logs/valid_predictions/epoch_{self.current_epoch}'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w+') as out_file:
                    for i, pred in enumerate(all_pred):
                        true = all_true[i]
                        out_file.write(f'{pred},{true}\n')

            results = self.get_evaluations(
                results, torch.cat(all_true),
                torch.cat(all_pred))

            self.valid_epoch_outputs = results
            # This log is used for monitoring metric and saving the best model. The actual logging happends within
            # clearml. See Monitor.py
            for key in results.keys():
                self.log(key, results[key])



        # results = {}
        # all_pred = [output['pred_y'] for output in valid_step_outputs]
        # all_true = [output['true_y'] for output in valid_step_outputs]
        #
        # # Store prediciton and labels if needed
        # if self.record_valid_pred:
        #     filename = f'logs/valid_predictions/epoch_{self.current_epoch}'
        #     os.makedirs(os.path.dirname(filename), exist_ok=True)
        #     with open(filename, 'w+') as out_file:
        #         for i, pred in enumerate(all_pred):
        #             true = all_true[i]
        #             out_file.write(f'{pred},{true}\n')
        #
        # results = self.get_evaluations(
        #     results, torch.cat(all_true),
        #     torch.cat(all_pred))
        #
        # self.valid_epoch_outputs = results
        # # This log is used for monitoring metric and saving the best model. The actual logging happends within
        # # clearml. See Monitor.py
        # for key in results.keys():
        #     self.log(key, results[key])
            
        # Logging
        # self.log(f"valid performance by epoch", self.valid_epoch_outputs, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch_data, batch_idx):
        """
        Process the data in validation dataloader in test mode
        :param batch_data:
        :param batch_idx:
        :return: It returns a list. The list is the outputs (a list) from
        the testing datasets. The item in the list is a dictionary
        from each step.
        """

        output = self(batch_data)
        pred_y = output[0].view(-1)
        true_y = batch_data.y.view(-1)
        # print(f'y_pred.shape:{y_pred.shape} y_true:{y_true.shape}')

        # Get numpy_prediction and numpy_y and concate those from all batches
        test_step_output = {}
        test_step_output['pred_y'] = pred_y
        test_step_output['true_y'] = true_y
        return test_step_output

    def test_epoch_end(self, test_step_outputs):
        """
        Evaluate on both the validation and training datasets. Besides in the
        training loop, the training dataset is included again because the
        model is set to evaluation mode (see
        https://stackoverflow.com/questions/60018578/what-does-model-eval-do
        -in-pytorch for a introduction of evaluation mode).
        :param test_step_outputs: a list of outputs from two dataloader.
        See the return description from function validation_step() above.
        set dataloader
        :return: None. However, set self.test_epoch_outputs to be a
        dictionary of metrics from each validation step, with metrics
        from training dataset with "_no_dropout" suffix, such as
        "loss_no_dropout". The self.valid_epoch_outputs is used for monitoring.
        """
        self.test_epoch_outputs = {}

        # There are true_y and pred_y from both validation and training
        # datasets from each validation iteration. Here we get the
        # concatenate them and calculate the metrics for all of them
        results = {}
        all_pred = torch.cat([output['pred_y'] for output in test_step_outputs])
        all_true = torch.cat([output['true_y'] for output in test_step_outputs])

        # Save pred and true in a file
        filename = 'logs/test_sample_scores.log'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as out_file:
            for i, pred in enumerate(all_pred):
                true = all_true[i]
                out_file.write(f'{pred},{true}\n')


        results = self.get_evaluations(
            results, all_true, all_pred)

        # Logging
        for key in results.keys():
            self.log(key, results[key])

        self.test_epoch_outputs = results

        # Logging
        # self.log(f"valid performance by epoch", self.valid_epoch_outputs,
        # on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """
        A required function for pytorch lightning class LightningModule.
        :return: A union of lists, the first one is optimizers and the
        second one is schedulers
        """
        optimizer = Adam(self.parameters())
        # scheduler = warmup.
        scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_iterations=self.warmup_iterations,
                tot_iterations=self.tot_iterations,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        # return optimizer, scheduler

        # optimizer, scheduler = self.gnn_model.configure_optimizers(
        #     self.warmup_iterations, self.tot_iterations, self.peak_lr,
        #     self.end_lr)
        return [optimizer], [scheduler]

    def save_atom_encoder(self, dir, file_name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        torch.save(self.gnn_model.atom_encoder.state_dict(), dir + file_name)

    def save_graph_embedding(self, dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        torch.save(self.graph_embedding, f'{dir}/graph_embedding.pt')
        with open(f'{dir}/smiles_for_graph_embedding.txt', 'w+') as f:
            for smiles in self.smiles_list:
                f.write(smiles + "\n")

    def save_kernels(self, dir, file_name):
        """
        Save the kernels. Unique for Kernel GNN
        :param file_name:
        :return:
        """
        if isinstance(self.gnn_model, KGNNNet):
            if not os.path.exists(dir):
                os.mkdir(dir)
            torch.save(self.gnn_model.gnn.layers[
                0].trainable_kernelconv_set.state_dict(),
                dir + file_name)
        else:
            raise Exception("model.py::GNNModel.sve_kernels(): only "
                            "implemented for Kernel GNN")

    def print_graph_embedding(self):
        print(f'model.py::graph_embedding:\n{self.graph_embedding}')

    @staticmethod
    def add_model_args(gnn_type, parent_parser):
        """
        Add model arguments to the parent parser
        :param gnn_type: a lowercase string specifying GNN type
        :param parent_parser: parent parser for adding arguments
        :return: parent parser with added arguments
        """
        parser = parent_parser.add_argument_group("GNN_Model")

        # Add general model arguments below
        # E.g., parser.add_argument('--general_model_args', type=int,
        # default=12)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--record_valid_pred', action='store_true',
                            default=False)
        parser.add_argument(f'--train_metric', action = 'store_true',
                            default=False)
        parser.add_argument('--warmup_iterations', type=int, default=60000)
        parser.add_argument('--peak_lr', type=float, default=5e-2)
        parser.add_argument('--end_lr', type=float, default=1e-9)

        # For linear layer
        parser.add_argument('--ffn_dropout_rate', type=float, default=0.25)
        parser.add_argument('--ffn_hidden_dim', type=int, default=64)
        parser.add_argument('--task_dim', type=int, default=1)

        if gnn_type == 'gcn':
            GCNNet.add_model_specific_args(parent_parser)
        elif gnn_type == 'chebnet':
            ChebNet.add_model_specific_args(parent_parser)
        elif gnn_type == 'kgnn':
            KGNNNet.add_model_specific_args(parent_parser)
        elif gnn_type == 'dimenet_pp':
            DimeNetPP.add_model_specific_args(parent_parser)
        elif gnn_type == 'chironet':
            ChIRoNet.add_model_specific_args(parent_parser)
        elif gnn_type == 'spherenet':
            SphereNet.add_model_specific_args(parent_parser)
        else:
            NotImplementedError('model.py::GNNModel::add_model_args(): '
                                'gnn_type is not defined for args groups')

        return parent_parser

    def get_evaluations(self, results, true_y, pred_y):
        '''
        :param results: a dictionary to store the result. Results will be
        appended to the existing dictionary.
        :param true_y: Tensor
        :param pred_y: Tensor
        :return:
        '''
        # Calculate loss using tensor
        loss = self.loss_func(pred_y, true_y.float())
        results['loss'] = loss

        # Convert tensor to numpy
        numpy_prediction = pred_y.detach().cpu().numpy()
        numpy_y = true_y.cpu().numpy()

        for metric in self.metrics:
            if metric == 'accuracy':
                accuracy = calculate_accuracy(numpy_y, numpy_prediction)
                results['accuracy'] = accuracy
                continue
            if metric == 'RMSE':
                rmse = mean_squared_error(numpy_y, numpy_prediction, squared=False)  # Setting
                # squared=False returns RMSE
                results['RMSE'] = rmse
            if metric == 'logAUC_0.001_0.1':
                logAUC = calculate_logAUC(numpy_y, numpy_prediction)
                results['logAUC_0.001_0.1'] = logAUC
            if metric == 'logAUC_0.001_1':
                logAUC = calculate_logAUC(numpy_y, numpy_prediction, FPR_range=(0.001, 1))
                results['logAUC_0.001_1'] = logAUC
            if metric == 'ppv':
                ppv = calculate_ppv(numpy_y, numpy_prediction)
                results['ppv'] = ppv
            if metric == 'f1_score':
                f1_sc = calculate_f1_score(numpy_y, numpy_prediction)
                results['f1_score'] = f1_sc
            if metric == 'AUC':
                AUC = calculate_auc(numpy_y, numpy_prediction)
                results['AUC'] = AUC
        return results
