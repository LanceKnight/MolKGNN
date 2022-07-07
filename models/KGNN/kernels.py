from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

import torch
from torch.nn import ModuleList, CosineSimilarity, Module, Linear, Sigmoid
# from torch.nn import
from torch.nn.parameter import Parameter

from itertools import permutations, combinations
import math
import pandas as pd

import time

import os

from .get_customized_kernels import get_hop1_kernel_list, \
    hop1_degree1_functional_groups, hop1_degree2_functional_groups, \
    hop1_degree3_functional_groups, hop1_degree4_functional_groups, \
    generate_kernel_with_angle_and_length_and_edge_attr

torch.autograd.set_detect_anomaly(True)


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
                 weight_requires_grad=False):
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
                                   x_support=torch.randn(L, num_supports,
                                                         node_attr_dim),
                                   edge_attr_support=torch.randn(L,
                                                                 num_supports,
                                                                 edge_attr_dim),
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
        # print(f'output:{output}')
        return output

    def intra_angle(self, input):
        """
        Calculate cosine of angles between all pairs of vectors stored in a
        tensor
        :param input: Shape[num_vectors, dim]
        :return: a tensor of Shape[num_combination, dim]
        """
        all_vecotr = input.unbind()
        num_vector = len(all_vecotr)

        # Get cosine angle between all paris of tensors in the input
        cos = CosineSimilarity(dim=-1)
        res_list = []
        all_combination = combinations(range(num_vector), 2)  # All
        for combination in all_combination:
            t1 = all_vecotr[combination[0]]
            t2 = all_vecotr[combination[1]]
            res_list.append(cos(t1, t2))
        result = torch.stack(res_list)
        result = torch.acos(result) # Convert to angle in radian
        return result

        # cos = CosineSimilarity(dim=-1)
        # new_p = torch.roll(p, 1, dims=-2)
        # sc = cos(new_p, p)
        # #         print(f'intra angle sc:{sc.shape}')
        # return sc

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
            if sim_dim > avg_dim:  # The sim_dim disappear after Cos,
                # so avg_dim
                # changes as well
                avg_dim = avg_dim - sim_dim
            sc = torch.mean(sc, dim=avg_dim)
        return sc

        # diff = torch.square(tensor1 - tensor2)
        # if dim is not None:
        #     sc = torch.sum(diff, dim=dim)
        # else:
        #     sc = torch.sum(diff)
        # sc = torch.atan(1 / (sc + 1e-8))
        return sc

    def get_angle_score(self, p_neighbor, p_support):
        """
        Calculate angle scores between the coordinates of neighbors and
        supports.

        It first calculates angles between each pairs of vectors in the
        neighbors/supports. Then these pairs of angles are compared between
        neighbors and supports

        :param p_neighbor: a tensor of size [num_nodes_of_this_degree, degree,
        space_dim] for the neighborhood
        :param p_support: a tensor of size [num_kernels,
        num_nodes_of_this_degree, degree, space_dim] for the kernel
        :return: returns the angle score, a tensor of Shape[num_kernels,
        num_nodes_of_this_degree]
        """

        # If the degree is 1, always return a tensor of max similarity (i.e.,1)
        deg = p_support.shape[-2]
        if ( deg == 1):
            num_kernel = p_support.shape[0]
            num_nodes_of_this_degree = p_neighbor.shape[0]
            return torch.full((num_kernel, num_nodes_of_this_degree),
                             1, device=p_neighbor.device)

        # # Debug
        # if deg == 2 or deg == 3 or deg == 4:
        #     print(f'======================')
        #     print(f'get_angle():p_neighbor:{p_neighbor}')
        #     print(f'get_angle():p_support:{p_support}')

        p_neighbor_for_all_node = p_neighbor.unbind()
        p_support_for_all_kernel = p_support.unbind()

        res_list_for_all_kernel = []
        for p_support_each_kernel in p_support_for_all_kernel:
            p_support_for_all_node = p_support_each_kernel.unbind()
            res_list_for_all_node = []
            for node_id, p_support_each_node in \
                    enumerate(p_support_for_all_node):
                support_intra_angle = self.intra_angle(p_support_each_node)
                # support_intra_angle = (support_intra_angle * 10).round() / 10
                # print(f'support_intra_angle:{support_intra_angle}')

                neighbor_intra_angle = self.intra_angle(
                    p_neighbor_for_all_node[node_id])
                # neighbor_intra_angle = (neighbor_intra_angle * 10).round()/10
                # print(f'neighbor_intra_angle:{neighbor_intra_angle}')
                intra_angle_similarity = \
                    self.calculate_average_similarity_score(
                        support_intra_angle, neighbor_intra_angle, sim_dim
                        =-1)
                res_list_for_all_node.append(intra_angle_similarity)
            res_list_for_all_kernel.append(res_list_for_all_node)
        result = torch.tensor(res_list_for_all_kernel, device =
        p_support.device)

        # # Debug
        # if deg == 2 or deg == 3 or deg == 4:
        #     print(f'get_angle():result:{result}')

        return result

        # p_neighbor = p_neighbor.unsqueeze(0).unsqueeze(0).expand(
        #     p_support.shape[0], p_support.shape[1], p_neighbor.shape[-3],
        #     p_neighbor.shape[-2], p_neighbor.shape[-1])
        #
        # intra_p_neighbor_angle = self.intra_angle(p_neighbor)
        #
        # intra_p_support_angle = self.intra_angle(p_support)
        #
        # sc = self.arctan_sc(intra_p_neighbor_angle,
        #                     intra_p_support_angle, sim_dim=(-1))
        #
        # # Debug
        # print(f'get_angle():sc:{sc.shape}')
        # return sc.squeeze(1)

    def get_length_score(self, p_neighbor, p_support):
        """
        Compare the length of the neighbors and supports

        It calculates the norm for all vectors in neighbors/supports first
        and then calculates similarities between those norms
        :param p_neighbor: Shape[num_nodes_in_this_degree, degree, dim]
        :param p_support: Shape[num_kernels, num_nodes_in_this_degree,
        degree, dim]
        :return: a tensor. Shape[num_kernel, num_nodes_in_this_degree]
        """
        len_p_neighbor = torch.norm(p_neighbor, dim=-1)
        len_p_support = torch.norm(p_support, dim=-1)



        # Round the length of neighbors to 0.1. E.g., 1.512 becomes 1.5
        # This eliminates the impact of small difference in lengths
        len_p_neighbor = (len_p_neighbor * 10).round()/10

        # Debug
        # deg = p_neighbor.shape[-2]
        # if deg == 4:
        #     print(f'get_length():p_neighbor:{p_neighbor}')
        #     print(f'get_length_score():p_neighbor:{p_neighbor.shape} p_support:'
        #           f'{p_support.shape}')
        #     print(f'get_length_score():\nlen_p_neighbor:\n'
        #           f'{len_p_neighbor}\nlen_p_support:\n{len_p_support}')

        # Get the similarity score
        sc = self.calculate_average_similarity_score(len_p_neighbor, len_p_support, sim_dim=(-1))
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
        # print('===start==')
        # print(f'before x_nei:{x_nei.shape} numel: '
        #       f'{torch.numel(x_nei)/1000000}M mem:'
        #       f'{self.mem_size(x_nei)/1000000} '
        #       f'MB')
        # print(f'before x_suppport:{x_support.shape} numel:'
        #       f' {torch.numel(x_support)/1000000}M mem:'
        #       f'{self.mem_size(x_support)/1000000}MB')



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


        # x_nei_node_list = x_nei.unbind()
        # x_support_kernel_list = x_support.unbind()
        # res_kernel = []
        # for x_support_kernel in x_support_kernel_list:
        #     x_support_permute_list = x_support_kernel.unbind()
        #     res_permute = []
        #     for x_support_permute in x_support_permute_list:
        #         def sim(input):
        #             sc = self.calculate_average_similarity_score(input,
        #                                                          x_support_permute,
        #                                                          sim_dim=-1,
        #                                                          avg_dim=-2)
        #             return sc
        #
        #         res_node = list(map(sim, x_nei_node_list))
        #         # for x_nei_node in x_nei_node_list:
        #
        #             # res_node.append(sc)
        #         res_permute.append(res_node)
        #     res_kernel.append(res_permute)
        # sc = torch.tensor(res_kernel, device=x_nei.device)
        # return sc








        # # ====================
        # # Debugging
        # # deg = x_support.shape[-2]
        # # if deg == 4:
        # #     print(f'kernels.py::\nx_nei:\n{x_nei}\nx_support:\n{x_support}')
        #
        # x_nei = x_nei.unsqueeze(0).unsqueeze(0).expand(
        #     x_support.shape[0], x_support.shape[1], x_nei.shape[0],
        #     x_nei.shape[1], x_nei.shape[2])
        # x_support = x_support.unsqueeze(2).expand(x_nei.shape)
        # sc = self.calculate_average_similarity_score(x_nei, x_support, sim_dim=-1, avg_dim=-2)
        # # print(f'kernels.py::sc:{sc.shape}')
        # # # =====================
        # #
        # # print(f'after x_nei shape:{x_nei.shape}, numel:'
        # #       f'{torch.numel(x_nei)/1000000}M mem:'
        # #       f'{self.mem_size(x_nei)/1000000}MB')
        # # print(f'x_support shape:{x_support.shape}, numel:'
        # #       f'{torch.numel(x_support)/1000000}M mem:'
        # #       f'{self.mem_size(x_support)/(1024*1024)}MB')
        # # print(f'sc shape:{sc.shape}')
        # # print('===end==')
        # return sc

    def get_center_attribute_score(self, x_focal, x_center):
        """
        Get the similarity score between center and focal atoms in the
        neighborhood and filter
        :param x_focal: Shape[num_node, node_attr_dim]
        :param x_center: Shape[num_kernels, node_attr_dim]
        :return: a tensor of Shape[num_kernels, num_node]
        """

        # x_focal_node_list = x_focal.unbind()
        # x_center_kernel_list = x_center.unbind()
        # res_kernel = []
        # for x_center_kernel in x_center_kernel_list:
        #     res_node = []
        #     for x_focal_node in x_focal_node_list:
        #         sc = self.calculate_average_similarity_score(x_focal_node,
        #                                                      x_center_kernel,
        #                                                 sim_dim=(-1))
        #         res_node.append(sc)
        #     res_kernel.append(res_node)
        # sc = torch.tensor(res_kernel, device = x_focal.device)

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
                    node_res_list.append([1]*num_kernel) # Return 1 for all
                    # kernels
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


    def calculate_total_score(self, x_focal, p_focal, x_neighbor, p_neighbor,
                              edge_attr_neighbor):
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
        # if deg == 4:
        #     print(f'total_score():deg:{deg}-------------------')
            # print(f'total_score():x_focal:{x_focal}')
            # print(f'total_score():x_neighbor:{x_neighbor}')

        # Because every sub-score is calculated using actan function,
        # which peaks at pi/2, so this max_atn is used to normalized the
        # score so it is in [0,1]
        # max_atan = torch.tensor([math.pi / 2], device=p_neighbor.device)

        # Calculate the support attribute score
        permuted_x_support = self.permute(x_support)
        support_attr_sc = self.get_support_attribute_score(x_neighbor,
                                                           permuted_x_support)

        # Get the best support_attr_sc and its index
        best_support_attr_sc, best_support_attr_sc_index = torch.max(
            support_attr_sc, dim=1)

        # # Debug
        # if deg==3:
        #     print(f'KernelConv()::deg={deg} p_neighbor:\n{p_neighbor}')

        # # Calculate position score
        # best_p_support = self.get_the_permutation_with_best_alignment_id(
        #     p_support, best_support_attr_sc_index)
        # position_sc = self.get_position_score(p_neighbor, best_p_support)

        # # Calculate the angle score
        best_p_support = self.get_the_permutation_with_best_alignment_id(
            p_support, best_support_attr_sc_index)
        # # permuted_p_support = self.permute(p_support)
        # # permuted_p_support = permuted_p_support.unsqueeze(2).expand(
        # #     permuted_p_support.shape[0], permuted_p_support.shape[1],
        # #     best_support_attr_sc_index.shape[1], permuted_p_support.shape[2],
        # #     permuted_p_support.shape[3])
        # # selected_index = best_support_attr_sc_index.unsqueeze(1).unsqueeze(
        # #     -1).unsqueeze(-1).expand(
        # #     permuted_p_support.shape[0], 1,
        # #     best_support_attr_sc_index.shape[-1],
        # #     permuted_p_support.shape[3],
        # #     permuted_p_support.shape[4])
        # # best_p_support = torch.gather(permuted_p_support, 1, selected_index)
        angle_sc = self.get_angle_score(p_neighbor, best_p_support)
        #
        # # print(f'best_p_support:{best_p_support}')
        #
        # # Calculate length score
        # best_p_support = best_p_support.squeeze(1)
        length_sc = self.get_length_score(p_neighbor,
                                          best_p_support)

        # Calculate the center attribute score
        center_attr_sc = self.get_center_attribute_score(x_focal,
                                                         x_center)


        # Calculate the edge attribute score
        selected_index = best_support_attr_sc_index.unsqueeze(-1).unsqueeze(
            -1).expand(
            best_support_attr_sc_index.shape[0],
            best_support_attr_sc_index.shape[1], edge_attr_support.shape[-2],
            edge_attr_support.shape[-1])
        permuted_edge_attr_support = self.permute(edge_attr_support)
        best_edge_attr_support = torch.gather(
            permuted_edge_attr_support, 1, selected_index)
        edge_attr_support_sc = self.get_edge_attribute_score(
            edge_attr_neighbor, best_edge_attr_support)
        support_attr_sc = best_support_attr_sc

        # Debug
        # if deg == 4:
        #     # print(f'best_support_attr_sc:{best_support_attr_sc}\n '
        #     #       f'best_support_attr_sc_index:{best_support_attr_sc_index}'
        #     #       f'\n ')
        #     print(f'best_position_sc:{position_sc}')

        # if deg == 4:
        #     start_chirality = time.time()
        #     chirality_sign = self.get_chirality_sign(p_neighbor,
        #                                              x_neighbor,
        #                                              best_p_support
        #                                              )
        #     # print(f'chirality sign: support_attr_sc:{support_attr_sc.shape}')
        #     # print(f'chirality sign: chirality_sign:{chirality_sign.shape}')
        #     support_attr_sc = support_attr_sc * chirality_sign
        #     end_chirality = time.time()
        #     print(f'=====kernels.py::chirality:{end_chirality-start_chirality}')


        # Debug
        # if (deg == 4):
        #     # print(f'kernels.py::length:{length_sc}')
        #     # print(f'kernels.py::angle:{angle_sc}')
        #     print(f'==============')
        #     print(f'kernels.py::support_attr_sc:{support_attr_sc}')
        #     print(f'kernels.py::center_attr_sc:{center_attr_sc}')
        #     print(f'kernels.py::edge_attr_support_sc:'
        #           f'{edge_attr_support_sc}')
        #     print(f'neighborhood:----')
        #     torch.set_printoptions(profile="full")
        #     print(f'x_focal:\n{x_focal}')
        #     print(f'kernels:----------')
        #     print(f'x_center:\n{x_center}')
        #     print(f'x_support:\n{x_support}')



        # Each score is of Shape[num_kernel, num_nodes_of_this_degree]
        sc = (
                 length_sc * self.length_sc_weight
                 + angle_sc * self.angle_sc_weight
                 + support_attr_sc * self.support_attr_sc_weight
                 + center_attr_sc * self.center_attr_sc_weight
                 + edge_attr_support_sc * self.edge_attr_support_sc_weight
                 # + position_sc * self.length_sc_weight
             ) / (self.support_attr_sc_weight+self.center_attr_sc_weight +
                  self.edge_attr_support_sc_weight + self.length_sc_weight + self.angle_sc_weight)
        b = time.time()
        return sc
        # return sc, length_sc, angle_sc, support_attr_sc, center_attr_sc, \
        #        edge_attr_support_sc

    def forward(self, *argv, **kwargv):
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

        # Check if neighborhood and kernel agrees in space dimension (i.e.,
        # 2D or 3D).
        if (p_focal.shape[-1] != self.p_support.shape[-1]):
            raise Exception(
                f'data coordinates is of {p_focal.shape[-1]}D, but the '
                f'kernel is {self.p_support.shape[-1]}D')

        # sc, length_sc, angle_sc, supp_attr_sc, center_attr_sc, \
        # edge_attr_support_sc
        sc = self.calculate_total_score(
            x_focal, p_focal, x_neighbor, p_neighbor, edge_attr_neighbor)

        # print('\n')
        # print(f'len sc:{length_sc}')
        # print(f'angle sc:{angle_sc}')
        # print(f'support attribute_sc:{supp_attr_sc}')
        # print(f'center_attr_sc:{center_attr_sc}')
        # print(f'edge attribute score:{edge_attr_support_sc}')
        # print(f'total sc: {sc.shape}')
        return sc  # , length_sc, angle_sc, supp_attr_sc, center_attr_sc,
        # edge_attr_support_sc


class BaseKernelSetConv(Module):
    # , trainable_kernelconv2=None, trainable_kernelconv3=None,
    # trainable_kernelconv4=None, ):
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

        # if (trainable_kernelconv1 is not None) and (trainable_kernelconv2
        # is not None) and (trainable_kernelconv3 is not None) and (
        # trainable_kernelconv4 is not None):
        #     self.trainable_kernelconv_set = ModuleList([
        #     trainable_kernelconv1, trainable_kernelconv2,
        #     trainable_kernelconv3, trainable_kernelconv4]
        #                                                )  # ,
        #                                                trainable_kernelconv2, trainable_kernelconv3, trainable_kernelconv4])
        #     self.num_trainable_kernel_list = [
        #     trainable_kernelconv1.get_num_kernels(),
        #     trainable_kernelconv2.get_num_kernels(),
        #                                       trainable_kernelconv3.get_num_kernels(), trainable_kernelconv4.get_num_kernels()]
        # else:
        #     self.trainable_kernelconv_set = ModuleList([])
        #     self.num_trainable_kernel_list = []

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

        # num of kernels for each degree, combining both fixed and trainable
        # kerenls
        self.num_kernel_list = []
        for i in range(4):
            num = 0
            if (self.num_fixed_kernel_list[i] is not None):
                num = self.num_fixed_kernel_list[i]
            if (self.num_trainable_kernel_list[i] is not None):
                num += self.num_trainable_kernel_list[i]
            self.num_kernel_list.append(num)

        # print(f'self.num_kernel_list:{self.num_kernel_list}')

    #         kernel_set = ModuleList(
    #             [KernelConv(D=D, num_supports=1, node_attr_dim =
    #             node_attr_dim, edge_attr_dim = edge_attr_dim),
    #              KernelConv(D=D, num_supports=2, node_attr_dim =
    #              node_attr_dim, edge_attr_dim = edge_attr_dim),
    #              KernelConv(D=D, num_supports=3, node_attr_dim =
    #              node_attr_dim, edge_attr_dim = edge_attr_dim),
    #              KernelConv(D=D, num_supports=4, node_attr_dim =
    #              node_attr_dim, edge_attr_dim = edge_attr_dim)
    #             ])

    def get_focal_nodes_of_degree(self, x, p, selected_index):
        '''
        outputs
        ori_x: a feature matrix that only contains rows (i.e. the center
        node) having certain degree
        ori_p: a position matrix that only contains rows (i.e. the center
        node) having certain degree
        '''
        start = time.time()
        x_focal = torch.index_select(input=x, dim=0, index=selected_index)
        end = time.time()
        # print(f'=====kernels.py::x_focal index_select:{end-start}')
        # p_focal = torch.index_select(input=p, dim=0, index=selected_index)

        return x_focal  # , p_focal

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

        # 		num_focal = len(focal_index)
        # #         print('center_index')
        # #         print(center_index)

        # 		nei_x_list = []
        # 		nei_p_list = []
        # 		nei_edge_attr_list = []
        # 		print(f'nei_index:{torch.squeeze(nei_index.T)}')
        # print(f'nei_index:{nei_index.shape}')
        # nei_x = torch.index_select(x, 0, torch.squeeze(nei_index.T))

        start = time.time()
        nei_x = torch.index_select(x, 0, nei_index)
        end = time.time()
        # print(f'=====kernels.py::nei index_select:{end-start}')
        nei_x = nei_x.reshape(-1, deg, nei_x.shape[-1])
        # print(f'nei_x:{nei_x.shape}')

        # nei_p = torch.index_select(p, 0, torch.squeeze(nei_index))
        # if deg == 2:
        #     print(f'deg2 before nei_p:{nei_p}')
        # nei_p = nei_p.reshape(-1, deg, nei_p.shape[-1])
        # if deg == 2:
        #     print(f'deg2 after nei_x:{nei_x}')
        #     print(f'deg2 after nei_p:{nei_p}')
        #             print(f'nei_p:{nei_p.shape}')

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
            #             print(f'x_neighbor:{x_neighbor.shape}')
            #             print(f'p_neighbor:{p_neighbor.shape}')
            # end = time.time()
            # print(f'2receptive_field:{end-start}')
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

    def forward(self, *argv, **kwargv):
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
        #         print('edge_index')
        #         print(edge_index)

        #         print('edge_attr')
        #         print(edge_attr)

        # loop through all possbile degrees. i.e. 1 to 4 bonds
        sc_list = []
        index_list = []

        # print(f'sum(self.num_kernel_list):{sum(self.num_kernel_list)}')
        zeros = torch.zeros(sum(self.num_kernel_list), x.shape[0],
                            device=p.device)
        # print('zeros')
        # print(zeros)
        start_row_id = 0
        start_col_id = 0
        for deg in range(1, 5):
            # print(f'deg:{deg}')
            start_deg = time.time()
            # x_focal = x_focal_list[deg-1]
            # p_focal = p_focal_list[deg-1]
            # x_neighbor = nei_x_list[deg-1]
            # p_neighbor = nei_p_list[deg-1]
            edge_attr_neighbor = nei_edge_attr_list[deg - 1]
            selected_index = selected_index_list[deg - 1]
            p_focal = p_focal_list[deg - 1]
            nei_index = nei_index_list[deg - 1]
            p_neighbor = nei_p_list[deg - 1]

            # if x_focal.shape[0] !=0: # make sure there are some nodes
            # having a certain degree
            # 	data = Data(x_focal=x_focal, p_focal=p_focal,
            # 	x_neighbor=x_neighbor, p_neighbor=p_neighbor,
            # 	edge_attr_neighbor=edge_attr_neighbor)
            start_convert = time.time()
            receptive_field = self.convert_graph_to_receptive_field(
                deg, x, p, edge_index, edge_attr,
                selected_index, nei_index
            )
            end_convert = time.time()
            # print(f'=====kernels.py::convert:{end_convert-start_convert}')
            # #             print('receptive_field')
            # #             print(receptive_field)
            if receptive_field is not None:
                x_focal, x_neighbor = receptive_field[0], receptive_field[1]
                data = Data(x_focal=x_focal, p_focal=p_focal,
                            x_neighbor=x_neighbor,
                            p_neighbor=p_neighbor,
                            edge_attr_neighbor=edge_attr_neighbor)

                # print(f'selected_index:{selected_index.shape}')
                # print('====data info====')
                # print('x_focal')
                # print(x_focal.shape)
                # print('p_focal')
                # print(p_focal.shape)
                # print('x_neighbor')
                # print(x_neighbor.shape)
                # print('p_neighbor')
                # print(p_neighbor.shape)
                # print('edge_attr_neighbor')
                # print(edge_attr_neighbor.shape)


                start_cal_sc = time.time()
                # Depanding on whether fixed kernels are used, choose the
                # correct KernelConv to use (either fixed_kernelConv,
                # trainable_kernel_conv, or both)
                if self.fixed_kernelconv_set[deg - 1] is not None:
                    fixed_degree_sc = self.fixed_kernelconv_set[deg - 1](
                        data=data)
                    if self.trainable_kernelconv_set[deg - 1] is not None:
                        trainable_degree_sc = self.trainable_kernelconv_set[
                            deg - 1](data=data)
                        degree_sc = torch.cat(
                            [fixed_degree_sc, trainable_degree_sc])
                    else:
                        degree_sc = fixed_degree_sc
                else:
                    if self.trainable_kernelconv_set[deg - 1] is not None:
                        trainable_degree_sc = self.trainable_kernelconv_set[
                            deg - 1](data=data)
                        degree_sc = trainable_degree_sc

                    else:
                        raise Exception(
                            f'kernels.py::BaseKernelSet:both fixed and '
                            f'trainable kernelconv_set are '
                            f'None for degree {deg}')
                # end_cal_sc = time.time()
                # print(f'====kernels.py::calc:{end_cal_sc-start_cal_sc}')

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
                #
                # if(deg == 4):
                #     torch.set_printoptions(profile="full")
                #     print(f'kernels.py::shape:{zeros.shape} zeros:{zeros}')
            else:
                start_row_id += self.num_kernel_list[deg - 1]
            # end_deg = time.time()
            # print(f'===kernels.py::deg_time:{end_deg-start_deg}')

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
        # end = time.time()
        # print(f'==kernels.py::kernelset:{end-start}')
        return sc


class KernelSetConv(BaseKernelSetConv):
    """
    Do the convolution on kernels of degree 1 to 4.
    """

    def __init__(self, L1, L2, L3, L4, D, node_attr_dim, edge_attr_dim):
        self.L = [L1, L2, L3, L4]

        # Test of std kernel
        p_support = torch.tensor([[1.2990e+00, 7.5000e-01]]).unsqueeze(0)
        # print(p_support)

        x_center = torch.tensor([[16, 32.067, 1.8, 2, 6]]).unsqueeze(0)

        x_support = torch.tensor(
            [[6.0000, 12.0110, 1.7000, 4.0000, 4.0000]]).unsqueeze(0)

        edge_attr_support = torch.tensor([[2]],
                                         dtype=torch.double).unsqueeze(0)

        kernel1_std = Data(p_support=p_support, x_support=x_support,
                           x_center=x_center,
                           edge_attr_support=edge_attr_support)

        p_support = torch.tensor([[1.2990e+00, 7.5000e-01],
                                  [-1.2990e+00, 7.5000e-01],
                                  [-2.7756e-16, -1.5000e+00]]).unsqueeze(0)
        # print(p_support)

        x_support = torch.tensor([[16, 32.067, 1.8, 2, 6],
                                  [6.0000, 12.0110, 1.7000, 4.0000, 4.0000],
                                  [1.0000, 1.0080, 1.2000, 1.0000,
                                   1.0000]]).unsqueeze(0)

        x_center = torch.tensor(
            [[6.0000, 12.0110, 1.7000, 4.0000, 4.0000]]).unsqueeze(0)

        edge_attr_support = torch.tensor(
            [[2], [1], [1]], dtype=torch.double).unsqueeze(0)

        kernel3_std = Data(p_support=p_support, x_support=x_support,
                           x_center=x_center,
                           edge_attr_support=edge_attr_support)

        #         kernel1 = KernelConv(init_kernel = kernel1_std)
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


class PredefinedKernelSetConv(BaseKernelSetConv):
    def __init__(self, D, node_attr_dim, edge_attr_dim, L1=0, L2=0, L3=0, L4=0,
                 is_first_layer=False):
        '''
        if is_first_layer == True, the use the fixed kernels, otherwise,
        don't use fixed kernels
        '''
        # generate functional kernels
        # trainable
        typical_smiles = 'C[H]'
        typical_center_atom_id = 1
        trainable_kernel1_list = []
        if L1 != 0:
            for i in range(L1):
                trainable_kernel1 = \
                    generate_kernel_with_angle_and_length_and_edge_attr(
                        D, typical_smiles, typical_center_atom_id,
                        node_attr_dim)
                trainable_kernel1_list.append(trainable_kernel1)
            self.trainable_kernel1 = self.cat_kernels(
                trainable_kernel1_list)  # generate a single tensor with L
            # as the first dimension from the list
            trainable_kernelconv1 = KernelConv(
                init_kernel=self.trainable_kernel1,
                requires_grad=True)  # generate the trainable KernelConv
        else:
            trainable_kernelconv1 = None
        # fixed
        if is_first_layer == True:
            fixed_kernel1_list = get_hop1_kernel_list(D)[0]
            self.fixed_kernel1 = self.cat_kernels(fixed_kernel1_list)
            fixed_kernelconv1 = KernelConv(init_kernel=self.fixed_kernel1,
                                           requires_grad=False)
            print(
                f'PredefinedKernelSetConv: there are '
                f'{self.fixed_kernel1.x_center.shape[0]} degree1 fixed '
                f'kernels, {L1} degree1 trainable kernels')
        else:
            print(
                f'PredefinedKernelSetConv: there are {L1} degree1 trainable '
                f'kernels')

        # degree2 kernels
        # trainable
        typical_smiles = 'CO[H]'
        typical_center_atom_id = 1
        trainable_kernel2_list = []
        if L2 != 0:
            for i in range(L2):
                trainable_kernel2 = \
                    generate_kernel_with_angle_and_length_and_edge_attr(
                        D, typical_smiles, typical_center_atom_id,
                        node_attr_dim)
                trainable_kernel2_list.append(trainable_kernel2)
            self.trainable_kernel2 = self.cat_kernels(
                trainable_kernel2_list)  # generate a single tensor with L
            # as the first dimension from the list
            trainable_kernelconv2 = KernelConv(
                init_kernel=self.trainable_kernel2,
                requires_grad=True)  # generate the trainable KernelConv
        else:
            trainable_kernelconv2 = None
        # fixed
        if is_first_layer == True:
            fixed_kernel2_list = get_hop1_kernel_list(D)[1]
            self.fixed_kernel2 = self.cat_kernels(fixed_kernel2_list)
            fixed_kernelconv2 = KernelConv(init_kernel=self.fixed_kernel2,
                                           requires_grad=False)
            print(
                f'PredefinedKernelSetConv: there are '
                f'{self.fixed_kernel2.x_center.shape[0]} degree2 fixed '
                f'kernels, {L2} degree2 trainable kernels')
        else:
            print(
                f'PredefinedKernelSetConv: there are {L2} degree2 trainable '
                f'kernels')

        # degree3 kernels
        # trainable
        typical_smiles = 'C=C'
        typical_center_atom_id = 1
        trainable_kernel3_list = []
        if L3 != 0:
            for i in range(L3):
                trainable_kernel3 = \
                    generate_kernel_with_angle_and_length_and_edge_attr(
                        D, typical_smiles, typical_center_atom_id,
                        node_attr_dim)
                trainable_kernel3_list.append(trainable_kernel3)
            self.trainable_kernel3 = self.cat_kernels(
                trainable_kernel3_list)  # generate a single tensor with L
            # as the first dimension from the list
            trainable_kernelconv3 = KernelConv(
                init_kernel=self.trainable_kernel3,
                requires_grad=True)  # generate the trainable KernelConv
        else:
            trainable_kernelconv3 = None
        # fixed
        if is_first_layer == True:
            fixed_kernel3_list = get_hop1_kernel_list(D)[2]
            self.fixed_kernel3 = self.cat_kernels(fixed_kernel3_list)
            fixed_kernelconv3 = KernelConv(init_kernel=self.fixed_kernel3,
                                           requires_grad=False)
            print(
                f'PredefinedKernelSetConv: there are '
                f'{self.fixed_kernel3.x_center.shape[0]} degree3 fixed '
                f'kernels, {L3} degree3 trainable kernels')
        else:
            print(
                f'PredefinedKernelSetConv: there are {L3} degree3 trainable '
                f'kernels')

        # degree4 kernels
        # trainable
        typical_smiles = 'CC'
        typical_center_atom_id = 1
        trainable_kernel4_list = []
        if L4 != 0:
            for i in range(L4):
                trainable_kernel4 = \
                    generate_kernel_with_angle_and_length_and_edge_attr(
                        D, typical_smiles, typical_center_atom_id,
                        node_attr_dim)
                trainable_kernel4_list.append(trainable_kernel4)
            self.trainable_kernel4 = self.cat_kernels(
                trainable_kernel4_list)  # generate a single tensor with L
            # as the first dimension from the list
            trainable_kernelconv4 = KernelConv(
                init_kernel=self.trainable_kernel4,
                requires_grad=True)  # generate the trainable KernelConv
        else:
            trainable_kernelconv4 = None
        # fixed
        if is_first_layer == True:
            fixed_kernel4_list = get_hop1_kernel_list(D)[3]
            self.fixed_kernel4 = self.cat_kernels(fixed_kernel4_list)
            fixed_kernelconv4 = KernelConv(init_kernel=self.fixed_kernel4,
                                           requires_grad=False)
            print(
                f'PredefinedKernelSetConv: there are '
                f'{self.fixed_kernel4.x_center.shape[0]} degree4 fixed '
                f'kernels, {L4} degree4 trainable kernels')
        else:
            print(
                f'PredefinedKernelSetConv: there are {L4} degree4 trainable '
                f'kernels')

        if is_first_layer == True:
            super(PredefinedKernelSetConv, self).__init__(fixed_kernelconv1,
                                                          fixed_kernelconv2,
                                                          fixed_kernelconv3,
                                                          fixed_kernelconv4,
                                                          trainable_kernelconv1,
                                                          trainable_kernelconv2,
                                                          trainable_kernelconv3,
                                                          trainable_kernelconv4)
        else:
            super(PredefinedKernelSetConv, self).__init__(
                trainable_kernelconv1=trainable_kernelconv1,
                trainable_kernelconv2=trainable_kernelconv2,
                trainable_kernelconv3=trainable_kernelconv3,
                trainable_kernelconv4=trainable_kernelconv4)

    def cat_kernels(self, kernel_list):
        x_center_list = [kernel.x_center for kernel in kernel_list]
        x_support_list = [kernel.x_support for kernel in kernel_list]
        p_support_list = [kernel.p_support for kernel in kernel_list]
        edge_attr_support_list = [kernel.edge_attr_support for kernel in
                                  kernel_list]

        # for x_center in x_center_list:
        #     print(x_center.shape)
        x_center = torch.cat(x_center_list)
        x_support = torch.cat(x_support_list)
        p_support = torch.cat(p_support_list)
        edge_attr_support = torch.cat(edge_attr_support_list)
        data = Data(x_center=x_center, x_support=x_support,
                    p_support=p_support, edge_attr_support=edge_attr_support)
        return data

    def get_num_kernel(self):
        num_kernel = 0
        if hasattr(self, 'trainable_kernel1'):
            num_kernel = self.trainable_kernel1.x_center.shape[0]
        if hasattr(self, 'trainable_kernel2'):
            num_kernel += self.trainable_kernel2.x_center.shape[0]
        if hasattr(self, 'trainable_kernel3'):
            num_kernel += self.trainable_kernel3.x_center.shape[0]
        if hasattr(self, 'trainable_kernel4'):
            num_kernel += self.trainable_kernel4.x_center.shape[0]

        if hasattr(self, 'fixed_kernel1'):
            num_kernel += self.fixed_kernel1.x_center.shape[0]
        if hasattr(self, 'fixed_kernel2'):
            num_kernel += self.fixed_kernel2.x_center.shape[0]
        if hasattr(self, 'fixed_kernel3'):
            num_kernel += self.fixed_kernel3.x_center.shape[0]
        if hasattr(self, 'fixed_kernel4'):
            num_kernel += self.fixed_kernel4.x_center.shape[0]
        # total_num = self.fixed_kernel1.x_center.shape[0] +
        # self.fixed_kernel2.x_center.shape[0] +
        # self.fixed_kernel3.x_center.shape[0] +
        # self.fixed_kernel4.x_center.shape[0] + num_trainable_kernel
        # print(f'total number kernels:{total_num}')
        return num_kernel


# class PredefinedNHopKernelSetConv(BaseKernelSetConv):
#     '''
#     The main difference between a PredefinedNHopKernelSetConv(abbreviated
#     as NHop for simplicity) and Predefined1HopKernelSetConv(abbreviated as
#     1Hop for simplicity)
#     is that 1HOP has some fixed kernels but NHop has all trainable but
#     predefined kernels.
#     '''
#     def __init__(self, D, node_attr_dim, edge_attr_dim, L1=0, L2=0, L3=0,
#     L4=0):

#         # generate functional kernels
#         # degree1 kernels
#         typical_smiles = 'C[H]'
#         typical_center_atom_id = 1
#         trainable_kernel1_list = []
#         if L1 != 0:
#             for i in range(L1):
#                 trainable_kernel1 =
#                 generate_kernel_with_angle_and_length_and_edge_attr(D,
#                 typical_smiles, typical_center_atom_id, node_attr_dim)
#                 trainable_kernel1_list.append(trainable_kernel1)
#             self.trainable_kernel1 = self.cat_kernels(
#             trainable_kernel1_list)  # generate a single tensor with L as
#             the first dimension from the list
#             trainable_kernelconv1 = KernelConv(
#             init_kernel=self.trainable_kernel1, requires_grad=True)  #
#             generate the trainable KernelConv
#         else:
#             trainable_kernelconv1 = None
#         print(f'PredefinedNHopKernelSetConv: there are {L1} degree1
#         trainable kernels')


#         # degree2 kernels
#         # degree2 kernels
#         typical_smiles = 'CO[H]'
#         typical_center_atom_id = 1
#         trainable_kernel2_list = []
#         if L2 != 0:
#             for i in range(L2):
#                 trainable_kernel2 =
#                 generate_kernel_with_angle_and_length_and_edge_attr(D,
#                 typical_smiles, typical_center_atom_id, node_attr_dim)
#                 trainable_kernel2_list.append(trainable_kernel2)
#             self.trainable_kernel2 = self.cat_kernels(
#             trainable_kernel2_list)  # generate a single tensor with L as
#             the first dimension from the list
#             trainable_kernelconv2 = KernelConv(
#             init_kernel=self.trainable_kernel2, requires_grad=True)  #
#             generate the trainable KernelConv
#         else:
#             trainable_kernelconv2 = None
#         print(f'PredefinedNHopKernelSetConv: there are {L2} degree2
#         trainable kernels')

#         # degree3 kernels
#         typical_smiles = 'C=C'
#         typical_center_atom_id = 1
#         trainable_kernel3_list = []
#         if L3 != 0:
#             for i in range(L3):
#                 trainable_kernel3 =
#                 generate_kernel_with_angle_and_length_and_edge_attr(D,
#                 typical_smiles, typical_center_atom_id, node_attr_dim)
#                 trainable_kernel3_list.append(trainable_kernel3)
#             self.trainable_kernel3 = self.cat_kernels(
#             trainable_kernel3_list)  # generate a single tensor with L as
#             the first dimension from the list
#             trainable_kernelconv3 = KernelConv(
#             init_kernel=self.trainable_kernel3, requires_grad=True)  #
#             generate the trainable KernelConv
#         else:
#             trainable_kernelconv3 = None
#         print(f'PredefinedNHopKernelSetConv: there are {L3} degree3
#         trainable kernels')

#         # degree4 kernels
#         typical_smiles = 'CC'
#         typical_center_atom_id = 1
#         trainable_kernel4_list = []
#         if L4 != 0:
#             for i in range(L4):
#                 trainable_kernel4 =
#                 generate_kernel_with_angle_and_length_and_edge_attr(D,
#                 typical_smiles, typical_center_atom_id, node_attr_dim)
#                 trainable_kernel4_list.append(trainable_kernel4)
#             self.trainable_kernel4 = self.cat_kernels(
#             trainable_kernel4_list)  # generate a single tensor with L as
#             the first dimension from the list
#             trainable_kernelconv4 = KernelConv(
#             init_kernel=self.trainable_kernel4, requires_grad=True)  #
#             generate the trainable KernelConv
#         else:
#             trainable_kernelconv4 = None
#         print(f'PredefinedNHopKernelSetConv: there are {L4} degree4
#         trainable kernels')

#         super(PredefinedNHopKernelSetConv, self).__init__(
#         trainable_kernelconv1=trainable_kernelconv1,
#         trainable_kernelconv2=trainable_kernelconv2,
#         trainable_kernelconv3=trainable_kernelconv3,
#         trainable_kernelconv4=trainable_kernelconv4)

#     def cat_kernels(self, kernel_list):
#         x_center_list = [kernel.x_center for kernel in kernel_list]
#         x_support_list = [kernel.x_support for kernel in kernel_list]
#         p_support_list = [kernel.p_support for kernel in kernel_list]
#         edge_attr_support_list = [kernel.edge_attr_support for kernel in
#         kernel_list]

#         # for x_center in x_center_list:
#         #     print(x_center.shape)
#         x_center = torch.cat(x_center_list)
#         x_support = torch.cat(x_support_list)
#         p_support = torch.cat(p_support_list)
#         edge_attr_support = torch.cat(edge_attr_support_list)
#         data = Data(x_center=x_center, x_support=x_support,
#         p_support=p_support, edge_attr_support=edge_attr_support)
#         return data

#     def get_num_kernel(self):
#         num_trainable_kernel = 0
#         if hasattr(self, 'trainable_kernel1'):
#             num_trainable_kernel = self.trainable_kernel1.x_center.shape[0]
#         if hasattr(self, 'trainable_kernel2'):
#             num_trainable_kernel += self.trainable_kernel2.x_center.shape[0]
#         if hasattr(self, 'trainable_kernel3'):
#             num_trainable_kernel += self.trainable_kernel3.x_center.shape[0]
#         if hasattr(self, 'trainable_kernel4'):
#             num_trainable_kernel += self.trainable_kernel4.x_center.shape[0]

#         total_num = num_trainable_kernel
#         # print(f'total number kernels:{total_num}')
#         return total_num

# class KernelLayer(Module):
#     '''
#         a wrapper of KernelSetConv for clear input/output dimension, inputs:
#         D: dimension
#         L: number of KernelConvSet

#         the output will be of dimension L1+L2+L3+L4
#     '''

#     def __init__(self, x_dim, p_dim, edge_dim, L1=None, L2=None, L3=None,
#     L4=None, predined_kernelsets=True):

#         super(KernelLayer, self).__init__()
#         if(predined_kernelsets == True):
#             self.conv = PredefinedKernelSetConv(D=p_dim,
#             node_attr_dim=x_dim, edge_attr_dim=edge_dim)
#         else:
#             if L1 is None or L2 is None or L3 is None or L4 is None:
#                 raise Exception('KernelLayer(): if predined_kernelsets is
#                 false, then L1-L4 needs to be specified')
#             self.conv = KernelSetConv(L1, L2, L3, L4, D=p_dim,
#             node_attr_dim=x_dim, edge_attr_dim=edge_dim)

#     def forward(self, data):
#         return self.conv(data=data)


if __name__ == "__main__":
    print('testing')
    # model = Predefined1HopKernelSetConv(D=2, node_attr_dim=5, edge_attr_dim=1,
    #                                     L1=2, L2=3, L3=4, L4=2)
    # model = KernelSetConv(D=2, node_attr_dim=5, edge_attr_dim=1, L1=15, L2=15,
    #                       L3=15, L4=15)
    # for param in model.parameters():
    #     print(param)
    # num = model.get_num_kernel()
    # print(num)
