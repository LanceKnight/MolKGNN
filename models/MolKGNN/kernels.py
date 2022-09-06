from itertools import permutations, combinations
import torch
from torch_geometric.data import Data
from torch.nn import ModuleList, CosineSimilarity, Module
from torch.nn.parameter import Parameter
import pandas as pd
import os

class KernelConv(Module):
    def __init__(self,
                 L=None,
                 D=None,
                 num_supports=None,
                 node_attr_dim=None,
                 edge_attr_dim=None,
                 init_kernel=None,
                 requires_grad=True,
                 init_length_sc_weight=0.2,
                 init_angle_sc_weight=0.2,
                 init_center_attr_sc_weight=0.2,
                 init_support_attr_sc_weight=0.2,
                 init_edge_attr_support_sc_weight=0.2,
                 weight_requires_grad=True):
        """
        Do the molecular convolution between a neighborhood and a kernel
        :param L:
        :param D:
        :param num_supports:
        :param node_attr_dim:
        :param edge_attr_dim:
        :param init_kernel: if not provided, create a random kernel
        :param requires_grad: if true, the kernel is trainable
        :param init_length_sc_weight: initial length score weight
        :param init_angle_sc_weight: initial angle score weight
        :param init_center_attr_sc_weight: initial center attr score weight
        :param init_support_attr_sc_weight: initial support attr score weight
        :param init_edge_attr_support_sc_weight: initial edge attr score weight
        :param weight_requires_grad: if true, the weights of subscores are
        trainable
        """
        super(KernelConv, self).__init__()
        if init_kernel is None:
            if (L is None) or (D is None) or (num_supports is None) or (
                    node_attr_dim is None) or (edge_attr_dim is None):
                raise Exception(
                    'either number of kernels L, convolution dimention D, '
                    'number of support num_supports or feature dimension '
                    'node_attr_dim is not specified')
            else:
                init_kernel = Data(x_center=torch.randn(L, node_attr_dim),
                                   x_support=torch.randn(L, num_supports, node_attr_dim),
                                   edge_attr_support=torch.randn(L, num_supports, edge_attr_dim),
                                   p_support=torch.randn(L, num_supports, D))

        self.num_kernels = init_kernel.x_center.shape[0]

        x_center_tensor = init_kernel.x_center
        self.x_center = Parameter(x_center_tensor, requires_grad=requires_grad)

        x_support_tensor = init_kernel.x_support
        self.x_support = Parameter(x_support_tensor,
                                   requires_grad=requires_grad)

        edge_attr_support_tensor = init_kernel.edge_attr_support
        self.edge_attr_support = Parameter(edge_attr_support_tensor,
                                           requires_grad=requires_grad)

        p_support_tensor = init_kernel.p_support
        self.p_support = Parameter(p_support_tensor,
                                   requires_grad=requires_grad)

        self.length_sc_weight = Parameter(torch.tensor(init_length_sc_weight),
                                          requires_grad=weight_requires_grad)
        self.angle_sc_weight = Parameter(torch.tensor(init_angle_sc_weight),
                                         requires_grad=weight_requires_grad)
        self.center_attr_sc_weight = Parameter(
            torch.tensor(init_center_attr_sc_weight),
            requires_grad=weight_requires_grad)
        self.support_attr_sc_weight = Parameter(
            torch.tensor(init_support_attr_sc_weight),
            requires_grad=weight_requires_grad)
        self.edge_attr_support_sc_weight = Parameter(
            torch.tensor(init_edge_attr_support_sc_weight),
            requires_grad=weight_requires_grad)

    def get_num_kernels(self):
        return self.num_kernels

    def permute(self, x):
        """
        Get the possible permutations given a set of neighbors in a kernel.

        For 1 neighbor, there is only 1 permutation
        For 2 neighbors, there are 2 permutations
        For 3 neighbors, there are 6 permutations
        For 4 neighbors, there are 24 permutations. However, if we consider
        chirality, only 12 of the permuations are allowed. See [1] for
        detailed explanation.

        References
        [1]Pattanaik, L., Ganea, O.-E., Coley, I., Jensen, K. F., Green, W. H.,
        and Coley, C. W., “Message Passing Networks for Molecules with
        Tetrahedral Chirality”, <i>arXiv e-prints</i>, 2020.
        :param x: the matrix containing the neighbor. Size is [num_kernels,
        degree, dim]
        :return: a tensor of size [num_kernels, num_permutations, degree,
        dimension]
        """
        degree = x.shape[1]
        if degree != 4:
            l = [x[:, torch.tensor(permute), :]
                 for permute in list(permutations(range(degree)))]
        else:
            d4_permutations = [(0, 1, 2, 3),
                               (0, 2, 3, 1),
                               (0, 3, 1, 2),
                               (1, 0, 3, 2),
                               (1, 2, 0, 3),
                               (1, 3, 2, 0),
                               (2, 0, 1, 3),
                               (2, 1, 3, 0),
                               (2, 3, 0, 1),
                               (3, 0, 2, 1),
                               (3, 1, 0, 2),
                               (3, 2, 1, 0)
                               ]
            l = [x[:, torch.tensor(permute), :]
                 for permute in d4_permutations]
        output = torch.stack(l, dim=1)
        return output

    # def intra_angle(self, input):
    #     """
    #     Calculate cosine of angles between all pairs of vectors stored in a
    #     tensor
    #     :param input: Shape[num_vectors, dim]
    #     :return: a tensor of Shape[num_combination, dim]
    #     """
    #     all_vecotr = input.unbind()
    #     num_vector = len(all_vecotr)
    #
    #     # Get cosine angle between all paris of tensors in the input
    #     cos = CosineSimilarity(dim=-1)
    #     res_list = []
    #     all_combination = combinations(range(num_vector), 2)  # All
    #     for combination in all_combination:
    #         t1 = all_vecotr[combination[0]]
    #         t2 = all_vecotr[combination[1]]
    #         res_list.append(cos(t1, t2))
    #     result = torch.stack(res_list)
    #     result = torch.acos(result) # Convert to angle in radian
    #     return result

    def calculate_average_similarity_score(self, tensor1, tensor2, sim_dim=None, avg_dim=None):
        """
        Calculate the similarity between two tensors.

        This similarity is both calculated using a CosineSimilarity and an
        average.
        E.g.
        t1 = torch.tensor([[[1, 2, 3], [3, 2, 1]], [[1, 2, 3], [3, 2, 1]]],
        dtype=torch.double) # 2*2*3 tensor
        t2 = torch.tensor([[[1, 2, 3], [3, 2, 1]], [[1, 2, 1], [1, 2, 1]]],
        dtype=torch.double) # 2*2*3 tensor
        if sim_dim=-1, avg_dim = -2,
        This will first calculate cos similarity along dim -1, and then
        average over dim -2 (original dim -2, not the dim after cos
        similarity).
        The result is tensor([1.000, 0.8729]) because the average of the two
        similarity scores are 1.000 and 0.9729 respectively

        :param tensor1: input1
        :param tensor2: input1
        :param sim_dim: the dimension along which similarity is calculated
        This dimension becomes 1 after calculation. The sim_dim has to be
        expressed as a negative interger (for the ease of implementation).
        :param avg_dim: the dimension along which an arithmetic average is
        calculated. The sim_dim has to be expressed as a negative integer (for
        the ease of implementation).
        :return: a tensor of average scores
        """
        if sim_dim >= 0 or (avg_dim is not None and avg_dim >= 0):
            raise NotImplementedError("kernels.py::arctan_sc(). Currently "
                                      "this function is implemented assuming "
                                      "sim_dim and avg_dim both are negative. "
                                      "Change the implementation if using "
                                      "positive dimension")

        cos = CosineSimilarity(dim=sim_dim)
        sc = cos(tensor1, tensor2)
        if avg_dim is not None:
            if sim_dim > avg_dim:  # The sim_dim disappear after Cos, so avg_dim changes as well
                avg_dim = avg_dim - sim_dim
            sc = torch.mean(sc, dim=avg_dim)
        return sc

    def get_the_permutation_with_best_alignment_id(self, input_tensor,
                                                   best_alignment_id):
        """
        Get the permutations and the input_tensor and choose the best one
        specified by the best_alignement_id
        :param input_tensor: an input to be permuted. Shape[num_kernel,
        degree, dim]
        :param best_alignment_idx: a tensor specifying the best alignment
        id. Shape[num_kernel, num_nodes_of_this_degree]
        :return: a tensor of shape [num_kernel, num_nodes_of_this_degree,
        degree, dim]
        """

        permuted_input = self.permute(input_tensor)  # After permutation,
        # Shape[num_kernel, num_permutation, degree, dim]
        alignment_for_all_kernels = permuted_input.unbind()  # A tuple,
        # with each element being permutations for one kernel.
        id_for_all_kernels = best_alignment_id.unbind()  # A tuple,
        # with each element being best permutation index for each kernel

        # Get the best alignment for each kernel and store in the res_list.
        res_list = []
        for i, alignment_each_kernel in enumerate(alignment_for_all_kernels):
            res_list.append(alignment_each_kernel[id_for_all_kernels[i], :,
                            :])

        result = torch.stack(
            res_list)  # convert list of kernels back to tensor
        return result

    def mem_size(self, ten):
        return ten.element_size() * ten.nelement()

    def get_support_attribute_score(self, x_nei, x_support):
        """

        :param x_nei: [num_nodes_of_this_degree, deg, attr_dim]
        :param x_support: Shape[num_kernel, num_permute, deg, attr_dim]
        :return: a tensor of Shape[num_kernels,
        num_permute, num_node_of_this_degree]
        """
        res_permute = []
        x_support_permute_list = x_support.unbind(dim=1)
        x_nei = x_nei.unsqueeze(0).expand(x_support.shape[0],
                                          x_nei.shape[0], x_nei.shape[1],
                                          x_nei.shape[2])
        for x_support_permute in x_support_permute_list:
            x_support_permute = x_support_permute.unsqueeze(1).expand(
                x_nei.shape)
            sc = self.calculate_average_similarity_score(x_nei,
                                                         x_support_permute,
                                                         sim_dim=-1,
                                                             avg_dim=-2)
            res_permute.append(sc)
        sc = torch.stack(res_permute, dim=1)
        return sc

    def get_center_attribute_score(self, x_focal, x_center):
        """
        Get the similarity score between center and focal atoms in the
        neighborhood and filter
        :param x_focal: Shape[num_node, node_attr_dim]
        :param x_center: Shape[num_kernels, node_attr_dim]
        :return: a tensor of Shape[num_kernels, num_node]
        """

        x_focal = x_focal.unsqueeze(0).expand(
            x_center.shape[0], x_focal.shape[0], x_focal.shape[1])
        x_center = x_center.unsqueeze(1).expand(x_focal.shape)

        sc = self.calculate_average_similarity_score(x_focal, x_center,
                                                     sim_dim=-1)

        return sc

    def get_edge_attribute_score(self, edge_attr_nei, edge_attr_support):
        sc = self.calculate_average_similarity_score(edge_attr_nei, edge_attr_support, sim_dim=-1,
                                                     avg_dim=-2)
        return sc



    def get_chirality_sign(self, p_nei, x_nei, p_support):
        """
        Calculate the sign for an atom with four neighbors using signed
        tetrahedral volume [1]. If all four neighbors have different
        attributes, the sign is either 1 or -1. If any two of the four
        neighbors have the same attributes, the sign is still 1 (no chirality)

        References:
        [1] Sliwoski G, Lowe EW, Butkiewicz M, Meiler J. BCL::EMAS--
        enantioselective molecular asymmetry descriptor for 3D-QSAR. Molecules.
         2012 Aug 20;17(8):9971-89. doi: 10.3390/molecules17089971. P
         MID: 22907158; PMCID: PMC3805266.
        :param p_nei: The neighbors' calibrated coordinates (centered at
        the origin). A tensor of Shape[num_nodes_of_this_degree, 4, space_dim]
        :param x_nei: The neighbor attributes. A tensor of Shape[
        num_nodes_of_this_degree, 4, dim]
        :param p_support: The filters' coordinates. A tensor of Shape[num_kernels,
        num_nodes_of_this_degree, 4, space_dim]
        :return: a tensor of Shape[num_kernel, num_nodes_of_this_degree].
        Element is 1
        or -1
        """

        num_kernel = p_support.shape[0]
        x_nei_node_list = x_nei.unbind()
        node_res_list = []
        for node_id, x_nei_node in enumerate(x_nei_node_list):
            neighbor_list = x_nei_node.unbind()
            is_chiral = True

            # If any two of the four attributes are the same, not chiral
            combination_list = combinations(range(4), 2)
            for combination in combination_list:
                neighbor1 = neighbor_list[combination[0]]
                neighbor2 = neighbor_list[combination[1]]
                if torch.equal(neighbor1, neighbor2):
                    node_res_list.append([1]*num_kernel) # Return 1 for all kernels
                    is_chiral = False
                    break

            if is_chiral:
                support_kernel_list = p_support.unbind()
                p_nei_node_list = p_nei.unbind()
                kernel_res_list = []
                for kernel_id, kernel in enumerate(support_kernel_list):
                    support_node_list = kernel.unbind()
                    support_list = support_node_list[node_id].unbind()
                    neighbor_list = p_nei_node_list[node_id].unbind()
                    t1_nei = neighbor_list[0]
                    t2_nei = neighbor_list[1]
                    t3_nei = neighbor_list[2]

                    t1_support = support_list[0]
                    t2_support = support_list[1]
                    t3_support = support_list[2]

                    # Calculate the tetrahedral volume sign
                    sign_nei = torch.sign(torch.dot(t3_nei, torch.cross(
                        t1_nei, t2_nei)))
                    sign_support = torch.sign(torch.dot(t3_support,
                                                        torch.cross(
                                                            t1_support,
                                                            t2_support)))

                    if sign_nei == sign_support:
                        kernel_res_list.append(1)
                    else:
                        kernel_res_list.append(-1)
                node_res_list.append(kernel_res_list)

        result = torch.tensor(node_res_list, device = x_nei.device).T
        return result


    def calculate_total_score(self, x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor, is_last_layer=False):
        # Calibrate neighbor coordinates
        # Calibrated coordinates = original coordinates - center coordinates
        p_neighbor = p_neighbor - p_focal.unsqueeze(1)

        # Get kernel params
        x_center = self.x_center
        x_support = self.x_support
        edge_attr_support = self.edge_attr_support
        p_support = self.p_support

        # Just for debugging
        deg = p_support.shape[-2]

        # Calculate the support attribute score
        permuted_x_support = self.permute(x_support)
        support_attr_sc = self.get_support_attribute_score(x_neighbor,
                                                           permuted_x_support)

        # Get the best support_attr_sc and its index
        best_support_attr_sc, best_support_attr_sc_index = torch.max(support_attr_sc, dim=1)

        # # Calculate the angle score
        best_p_support = self.get_the_permutation_with_best_alignment_id(p_support, best_support_attr_sc_index)

        # Calculate the center attribute score
        center_attr_sc = self.get_center_attribute_score(x_focal, x_center)

        # Calculate the edge attribute score
        selected_index = best_support_attr_sc_index.unsqueeze(-1).unsqueeze(-1).expand(
            best_support_attr_sc_index.shape[0],
            best_support_attr_sc_index.shape[1], edge_attr_support.shape[-2],
            edge_attr_support.shape[-1])
        permuted_edge_attr_support = self.permute(edge_attr_support)
        best_edge_attr_support = torch.gather(
            permuted_edge_attr_support, 1, selected_index)
        edge_attr_support_sc = self.get_edge_attribute_score(
            edge_attr_neighbor, best_edge_attr_support)
        support_attr_sc = best_support_attr_sc


        # Calculation of chirality
        chirality_sign = 1
        if (deg == 4) and (is_last_layer):
            chirality_sign = self.get_chirality_sign(p_neighbor,
                                                     x_neighbor,
                                                     best_p_support
                                                     )

        exp_support_attr_weight = torch.exp(self.support_attr_sc_weight)
        exp_center_attr_weight = torch.exp(self.center_attr_sc_weight)
        exp_edge_attr_support_weight = torch.exp(self.edge_attr_support_sc_weight)

        denominator =  exp_support_attr_weight\
                      + exp_center_attr_weight\
                      + exp_edge_attr_support_weight

        support_attr_sc_weight = exp_support_attr_weight/denominator
        center_attr_sc_weight = exp_center_attr_weight/denominator
        edge_attr_support_sc_weight = exp_edge_attr_support_weight/denominator


        # Each score is of Shape[num_kernel, num_nodes_of_this_degree]
        sc = (

                 support_attr_sc * support_attr_sc_weight
                 + center_attr_sc * center_attr_sc_weight
                 + edge_attr_support_sc * edge_attr_support_sc_weight
             ) / (support_attr_sc_weight+center_attr_sc_weight +
                  edge_attr_support_sc_weight)
        if deg ==4:
            sc = sc * chirality_sign
        return sc


    def forward(self, is_last_layer, **kwargv):
        if len(kwargv) == 1:
            x_focal = kwargv['data'].x_focal
            p_focal = kwargv['data'].p_focal
            x_neighbor = kwargv['data'].x_neighbor
            p_neighbor = kwargv['data'].p_neighbor
            edge_attr_neighbor = kwargv['data'].edge_attr_neighbor
        else:
            x_focal = kwargv['x_focal']
            p_focal = kwargv['p_focal']
            x_neighbor = kwargv['x_neighbor']
            p_neighbor = kwargv['p_neighbor']
            edge_attr_neighbor = kwargv['edge_attr_neighbor']

        # Check if neighborhood and kernel agrees in space dimension (i.e., 2D or 3D).
        if (p_focal.shape[-1] != self.p_support.shape[-1]):
            raise Exception(
                f'data coordinates is of {p_focal.shape[-1]}D, but the kernel is {self.p_support.shape[-1]}D')

        sc = self.calculate_total_score(x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor, is_last_layer)
        return sc


class BaseKernelSetConv(Module):
    def __init__(self, fixed_kernelconv1=None, fixed_kernelconv2=None,
                 fixed_kernelconv3=None, fixed_kernelconv4=None,
                 trainable_kernelconv1=None, trainable_kernelconv2=None,
                 trainable_kernelconv3=None, trainable_kernelconv4=None):
        super(BaseKernelSetConv, self).__init__()

        self.fixed_kernelconv_set = ModuleList(
            [fixed_kernelconv1, fixed_kernelconv2, fixed_kernelconv3,
             fixed_kernelconv4])
        self.num_fixed_kernel_list = []
        if (fixed_kernelconv1 is not None):
            self.num_fixed_kernel_list.append(
                fixed_kernelconv1.get_num_kernels())
        else:
            self.num_fixed_kernel_list.append(None)
        if (fixed_kernelconv2 is not None):
            self.num_fixed_kernel_list.append(
                fixed_kernelconv2.get_num_kernels())
        else:
            self.num_fixed_kernel_list.append(None)
        if (fixed_kernelconv3 is not None):
            self.num_fixed_kernel_list.append(
                fixed_kernelconv3.get_num_kernels())
        else:
            self.num_fixed_kernel_list.append(None)
        if (fixed_kernelconv4 is not None):
            self.num_fixed_kernel_list.append(
                fixed_kernelconv4.get_num_kernels())
        else:
            self.num_fixed_kernel_list.append(None)

        self.trainable_kernelconv_set = ModuleList(
            [trainable_kernelconv1, trainable_kernelconv2,
             trainable_kernelconv3, trainable_kernelconv4])
        self.num_trainable_kernel_list = []
        if (trainable_kernelconv1 is not None):
            self.num_trainable_kernel_list.append(
                trainable_kernelconv1.get_num_kernels())
        else:
            self.num_trainable_kernel_list.append(None)
        if (trainable_kernelconv2 is not None):
            self.num_trainable_kernel_list.append(
                trainable_kernelconv2.get_num_kernels())
        else:
            self.num_trainable_kernel_list.append(None)
        if (trainable_kernelconv3 is not None):
            self.num_trainable_kernel_list.append(
                trainable_kernelconv3.get_num_kernels())
        else:
            self.num_trainable_kernel_list.append(None)
        if (trainable_kernelconv4 is not None):
            self.num_trainable_kernel_list.append(
                trainable_kernelconv4.get_num_kernels())
        else:
            self.num_trainable_kernel_list.append(None)

        # Num of kernels for each degree, combining both fixed and trainable kerenls
        self.num_kernel_list = []
        for i in range(4):
            num = 0
            if (self.num_fixed_kernel_list[i] is not None):
                num = self.num_fixed_kernel_list[i]
            if (self.num_trainable_kernel_list[i] is not None):
                num += self.num_trainable_kernel_list[i]
            self.num_kernel_list.append(num)


    def get_focal_nodes_of_degree(self, x, p, selected_index):
        '''
        outputs
        ori_x: a feature matrix that only contains rows (i.e. the center
        node) having certain degree
        ori_p: a position matrix that only contains rows (i.e. the center
        node) having certain degree
        '''
        x_focal = torch.index_select(input=x, dim=0, index=selected_index)
        return x_focal

    def get_neighbor_nodes_and_edges_of_degree(self, deg, x, p, nei_index):
        '''
        inputs:
        deg: the query degree
        num_focal: the number of focal nodes of degree deg in the graph

        outputs:
        nei_x: a feature matrix that only contains rows (i.e. the
        neighboring node) that its center node has certain degree
        nei_p: a position matrix that only contains rows (i.e. the
        neighboring node) that its center node has certain degree
        '''

        nei_x = torch.index_select(x, 0, nei_index)

        nei_x = nei_x.reshape(-1, deg, nei_x.shape[-1])


        return nei_x

    def convert_graph_to_receptive_field(self, deg, x, p, edge_index,
                                         edge_attr, selected_index, nei_index):
        """
        Convert a graph into receptive fields for a certain degree. Return
        None if there is no nodes with that degree.

        :param deg:
        :param x:
        :param p:
        :param edge_index:
        :param edge_attr:
        :param selected_index:
        :param nei_index:
        :return:
        """

        x_focal = self.get_focal_nodes_of_degree(
            x=x, p=p, selected_index=selected_index)

        num_focal = x_focal.shape[0]

        if num_focal != 0:
            x_neighbor = self.get_neighbor_nodes_and_edges_of_degree(deg=deg,
                                                                     x=x, p=p,
                                                                     nei_index=nei_index)
            return x_focal, x_neighbor
        return None

    def get_reorder_index(self, index):
        '''
        get the index to rearrange output score matrix so that it corespond
        to the order in the original x matrix

        '''
        rearranged, new_index = torch.sort(index, dim=0)
        return new_index

    def format_output(self, output):
        '''
        change the shape of output from (L, num_nodes, 4) to (num_nodes, 4*L)
        '''
        a = [output[i, :, :] for i in range(output.shape[0])]
        return torch.cat(a, dim=1)

    def save_score(self, sc):
        root = 'customized_kernels'
        print('saving score...')
        sc_np = sc.cpu().detach().numpy()
        files = os.listdir(root)
        headers = []
        for i, file in enumerate(files):
            names = list(pd.read_csv(root + '/' + file)['name'])
            headers += names
            rand_names = ['std_kernel'] * self.num_trainable_kernel_list[i]
            headers += rand_names
        print(headers)
        sc_df = pd.DataFrame(sc_np, columns=headers)
        sc_df = sc_df.transpose()
        sc_df.to_csv('scores.csv')

    def forward(self, is_last_layer, *argv, **kwargv):
        '''
        inputs:
        data: graph data containing feature matrix, adjacency matrix,
        edge_attr matrix
        '''
        # start = time.time()
        if len(argv) != 0:
            raise Exception(
                'Kernel does not take positional argument, use keyword '
                'argument instead. e.g. model(data=data)')

        if len(kwargv) == 2:
            x = kwargv['data'].x
            edge_index = kwargv['data'].edge_index
            edge_attr = kwargv['data'].edge_attr
            p = kwargv['data'].p

            p_focal_list = [kwargv['data'].p_focal_deg1,
                            kwargv['data'].p_focal_deg2,
                            kwargv['data'].p_focal_deg3,
                            kwargv['data'].p_focal_deg4]
            nei_p_list = [kwargv['data'].nei_p_deg1, kwargv['data'].nei_p_deg2,
                          kwargv['data'].nei_p_deg3, kwargv['data'].nei_p_deg4]
            nei_edge_attr_list = [kwargv['data'].nei_edge_attr_deg1,
                                  kwargv['data'].nei_edge_attr_deg2,
                                  kwargv['data'].nei_edge_attr_deg3,
                                  kwargv['data'].nei_edge_attr_deg4]
            selected_index_list = [kwargv['data'].selected_index_deg1,
                                   kwargv['data'].selected_index_deg2,
                                   kwargv['data'].selected_index_deg3,
                                   kwargv['data'].selected_index_deg4]
            nei_index_list = [kwargv['data'].nei_index_deg1,
                              kwargv['data'].nei_index_deg2,
                              kwargv['data'].nei_index_deg3,
                              kwargv['data'].nei_index_deg4]
            save_score = kwargv['save_score']

        else:
            x = kwargv['x']
            edge_index = kwargv['edge_index']
            edge_attr = kwargv['edge_attr']
            p = kwargv['p']

            p_focal_list = [kwargv['p_focal_deg1'], kwargv['p_focal_deg2'],
                            kwargv['p_focal_deg3'], kwargv['p_focal_deg4']]
            nei_p_list = [kwargv['nei_p_deg1'], kwargv['nei_p_deg2'],
                          kwargv['nei_p_deg3'], kwargv['nei_p_deg4']]
            nei_edge_attr_list = [kwargv['nei_edge_attr_deg1'],
                                  kwargv['nei_edge_attr_deg2'],
                                  kwargv['nei_edge_attr_deg3'],
                                  kwargv['nei_edge_attr_deg4']]
            selected_index_list = [kwargv['selected_index_deg1'],
                                   kwargv['selected_index_deg2'],
                                   kwargv['selected_index_deg3'],
                                   kwargv['selected_index_deg4']]
            nei_index_list = [kwargv['nei_index_deg1'],
                              kwargv['nei_index_deg2'],
                              kwargv['nei_index_deg3'],
                              kwargv['nei_index_deg4']]

            save_score = kwargv['save_score']

        index_list = []
        zeros = torch.zeros(sum(self.num_kernel_list), x.shape[0],
                            device=p.device)
        start_row_id = 0
        start_col_id = 0
        for deg in range(1, 5):
            edge_attr_neighbor = nei_edge_attr_list[deg - 1]
            selected_index = selected_index_list[deg - 1]
            p_focal = p_focal_list[deg - 1]
            nei_index = nei_index_list[deg - 1]
            p_neighbor = nei_p_list[deg - 1]


            receptive_field = self.convert_graph_to_receptive_field(
                deg, x, p, edge_index, edge_attr,
                selected_index, nei_index
            )

            if receptive_field is not None:
                x_focal, x_neighbor = receptive_field[0], receptive_field[1]
                data = Data(x_focal=x_focal, p_focal=p_focal,
                            x_neighbor=x_neighbor,
                            p_neighbor=p_neighbor,
                            edge_attr_neighbor=edge_attr_neighbor)


                # Depanding on whether fixed kernels are used, choose the
                # correct KernelConv to use (either fixed_kernelConv,
                # trainable_kernel_conv, or both)
                if self.fixed_kernelconv_set[deg - 1] is not None:
                    fixed_degree_sc = self.fixed_kernelconv_set[deg - 1](is_last_layer = is_last_layer, data=data )
                    if self.trainable_kernelconv_set[deg - 1] is not None:
                        trainable_degree_sc = self.trainable_kernelconv_set[deg - 1](is_last_layer = is_last_layer,
                                                                                     data=data)
                        degree_sc = torch.cat(
                            [fixed_degree_sc, trainable_degree_sc])
                    else:
                        degree_sc = fixed_degree_sc
                else:
                    if self.trainable_kernelconv_set[deg - 1] is not None:
                        trainable_degree_sc = self.trainable_kernelconv_set[
                            deg - 1](is_last_layer = is_last_layer, data=data)
                        degree_sc = trainable_degree_sc

                    else:
                        raise Exception(
                            f'kernels.py::BaseKernelSet:both fixed and '
                            f'trainable kernelconv_set are '
                            f'None for degree {deg}')

                # Fill a zero tensor will score for each degree in
                # corresponding positions
                zeros[
                start_row_id:start_row_id + self.num_kernel_list[deg - 1],
                start_col_id:start_col_id + x_focal.shape[0]] = degree_sc

                # Update the start_row_id and start_col_id so that next
                # iteration can use those to find correct position for
                # filling the zero tensor with score for each degree
                index_list.append(selected_index)
                start_row_id += self.num_kernel_list[deg - 1]
                start_col_id += x_focal.shape[0]

            else:
                start_row_id += self.num_kernel_list[deg - 1]


        # Reorder the output score tensor so that its rows correspond to the
        # original index in the feature matrix x. Score tensor has shape
        # [num_nodes, num_total_kernels]
        sc = zeros
        index_list = torch.cat(index_list)
        new_index = self.get_reorder_index(index_list)
        sc = sc[:, new_index]
        sc = sc.T

        if (save_score == True):
            self.save_score(sc)  # save scores for analysis
        return sc


class KernelSetConv(BaseKernelSetConv):
    """
    Do the convolution on kernels of degree 1 to 4.
    """

    def __init__(self, L1, L2, L3, L4, D, node_attr_dim, edge_attr_dim):
        self.L = [L1, L2, L3, L4]

        kernelconv1 = KernelConv(L=L1, D=D, num_supports=1,
                                 node_attr_dim=node_attr_dim,
                                 edge_attr_dim=edge_attr_dim)
        kernelconv2 = KernelConv(L=L2, D=D, num_supports=2,
                                 node_attr_dim=node_attr_dim,
                                 edge_attr_dim=edge_attr_dim)

        kernelconv3 = KernelConv(L=L3, D=D, num_supports=3,
                                 node_attr_dim=node_attr_dim,
                                 edge_attr_dim=edge_attr_dim)
        kernelconv4 = KernelConv(L=L4, D=D, num_supports=4,
                                 node_attr_dim=node_attr_dim,
                                 edge_attr_dim=edge_attr_dim)
        super(KernelSetConv, self).__init__(trainable_kernelconv1=kernelconv1,
                                            trainable_kernelconv2=kernelconv2,
                                            trainable_kernelconv3=kernelconv3,
                                            trainable_kernelconv4=kernelconv4)

    def get_num_kernel(self):
        return sum(self.L)


if __name__ == "__main__":
    print('testing')

