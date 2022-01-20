import rdkit
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
from random import randint
import networkx as nx
from wrapper import smiles2graph, get_atom_rep


def randomly_add_node(data):
    old_graph = to_networkx(data)
    old_nodes = old_graph.nodes
    num_old_nodes = len(old_nodes)
    randn = randint(0, num_old_nodes - 1)

    new_node = torch.tensor(get_atom_rep(6))
    # print(new_node.unsqueeze(-1).shape)
    # print(data.x.shape)
    x = torch.cat((data.x, new_node.unsqueeze(0)), dim=0)

    new_edge = torch.tensor([[randn, num_old_nodes], [num_old_nodes, randn]])
    edge_index = torch.cat((data.edge_index, new_edge), dim=1)
    # print(edge_index)

    new_edge_attr = torch.tensor([[1], [1]])
    edge_attr = torch.cat((data.edge_attr, new_edge_attr), dim=0)

    new_p = torch.tensor([0, 0]).unsqueeze(0)
    p = torch.cat((data.p, new_p), dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, p=p)
    # print(data)
    return data


def generate_2D_molecule_from_reference(smiles, num):
    '''generate molecules with similar connectivity with the reference molecule
    smiles: input molecule
    num: number of augmented molecules to generate
    '''
    data = smiles2graph(2, smiles)
    output_list = []
    for i in range(num):
        new_mol = randomly_add_node(data)
        output_list.append(new_mol)
    return output_list


if __name__ == '__main__':
    smi1 = "C1(=CC=CC(=C1)C(CC)C)O"
    smi2 = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    output = generate_2D_molecule_from_reference(smi1, 50)
    for i in range(len(output)):
        print(output)
