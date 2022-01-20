import torch

from torch_geometric.data import Data

from rdkit import Chem
import rdkit
# from rdkit import Chem
# # from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

import pandas as pd
import os

import random

from .loader import get_atom_rep


def generate_1hop_kernel(D, typical_compound_smiles, center_atom_id,
                         x_dim=None):
    #     '''
    #     given a typical compound containing a certain kernal, and the
    #     center atom id, genrate the kernel
    #     '''
    if D == None:
        raise Exception(
            'generate_kernel2grpah() needs to input D to specifiy 2D or 3D '
            'graph generation.')

    smiles = typical_compound_smiles.replace(r'/=', '=')
    smiles = typical_compound_smiles.replace(r'\=', '=')

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    # mol.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol)

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    all_atoms = mol.GetAtoms()
    center_atom = all_atoms[center_atom_id]
    # print(f'center atom:{center_atom.GetSymbol()}')

    atom_pos = []
    atom_attr = []

    supports = center_atom.GetNeighbors()

    if x_dim is None:
        x_center = get_atom_rep(center_atom.GetAtomicNum())
    else:
        x_center = [random.uniform(0, 1) for i in range(x_dim)]

    p_list = []
    x_list = []
    bond_attr_list = []
    # print()
    # print('atom idx:')
    # for i, atom in enumerate(all_atoms):
    #     print(f'{atom.GetIdx()}, {atom.GetSymbol()}')

    for idx, edge in enumerate(center_atom.GetBonds()):
        support_start_id = edge.GetBeginAtomIdx()
        support_end_id = edge.GetEndAtomIdx()
        #         print(f'support_start_id:{support_start_id}')
        #         print(f'support_end_id:{support_end_id}')
        if (support_start_id == center_atom_id):
            support_id = support_end_id
        else:
            support_id = support_start_id
        support = all_atoms[support_id]
        if x_dim is None:
            x_list.append(get_atom_rep(support.GetAtomicNum()))
        else:
            x_list.append([random.uniform(0, 1) for i in range(x_dim)])
        if D == 2:
            p_support = p_list.append([conf.GetAtomPosition(
                support_id).x - conf.GetAtomPosition(center_atom_id).x,
                                       conf.GetAtomPosition(
                                           support_id).y -
                                       conf.GetAtomPosition(
                                           center_atom_id).y])
        if D == 3:
            p_support = p_list.append([conf.GetAtomPosition(
                support_id).x - conf.GetAtomPosition(center_atom_id).x,
                                       conf.GetAtomPosition(support_id).y -
                                       conf.GetAtomPosition(center_atom_id).y,
                                       conf.GetAtomPosition(
                                           support_id).z -
                                       conf.GetAtomPosition(
                                           center_atom_id).z])

        bond_attr = None
        bond_type = edge.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_attr = [1]
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_attr = [2]
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            bond_attr = [3]
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_attr = [4]
        bond_attr_list.append(bond_attr)

    x_center = torch.tensor(x_center).unsqueeze(0).unsqueeze(0)
    x_support = torch.tensor(x_list).unsqueeze(0)
    p_support = torch.tensor(p_list).unsqueeze(0)
    edge_attr_support = torch.tensor(bond_attr_list,
                                     dtype=p_support.dtype).unsqueeze(0)

    #     print('x_center')
    #     print(x_center)
    #     print('x_support')
    #     print(x_support)
    #     print('p_support')
    #     print(p_support)
    #     print('edge_attr_support')
    #     print(edge_attr_support)
    data = Data(x_center=x_center, x_support=x_support, p_support=p_support,
                edge_attr_support=edge_attr_support)
    return data  # x_center, x_support, p_support, edge_attr_support


def generate_kernel_with_angle_and_length_and_edge_attr(D,
                                                        typical_compound_smiles,
                                                        center_atom_id, x_dim):
    '''
    generate a kernel with typical angle and lenth and edge_attr,
    but randomize x attribute
    '''
    return generate_1hop_kernel(D, typical_compound_smiles, center_atom_id,
                                x_dim=x_dim)


def read_kernel_from_csv(path):
    df = pd.read_csv(path, index_col=0)
    df = df.transpose()
    kernel_dict = df.to_dict(orient='list')
    return kernel_dict


def print_kernel_files():
    root = 'customized_kernels'
    files = os.listdir(root)
    for file in files:
        df = pd.read_csv(root + '/' + file)
        print(df)


# Get current file directory
current_directory = os.path.dirname(__file__)

# Get degree customized kernels defined in corresponding .csv files
hop1_degree1_functional_groups = read_kernel_from_csv(os.path.join(
    current_directory, 'customized_kernels/customized_kernel1.csv'))

hop1_degree2_functional_groups = read_kernel_from_csv(os.path.join(
    current_directory, 'customized_kernels/customized_kernel2.csv'))

hop1_degree3_functional_groups = read_kernel_from_csv(os.path.join(
    current_directory, 'customized_kernels/customized_kernel3.csv'))

hop1_degree4_functional_groups = read_kernel_from_csv(os.path.join(
    current_directory, 'customized_kernels/customized_kernel4.csv'))

# ===1hop kernels - 2D===
# degree1
functional_groups = hop1_degree1_functional_groups
hop1_2D_degree1_kernels_list = [
    generate_1hop_kernel(2, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]
# degree2
functional_groups = hop1_degree2_functional_groups
hop1_2D_degree2_kernels_list = [
    generate_1hop_kernel(2, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]
# degree3
functional_groups = hop1_degree3_functional_groups
hop1_2D_degree3_kernels_list = [
    generate_1hop_kernel(2, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]
# degree4
functional_groups = hop1_degree4_functional_groups
hop1_2D_degree4_kernels_list = [
    generate_1hop_kernel(2, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]

# ===1hop kernels - 3D===
# degree1
functional_groups = hop1_degree1_functional_groups
hop1_3D_degree1_kernels_list = [
    generate_1hop_kernel(3, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]
# degree2
functional_groups = hop1_degree2_functional_groups
hop1_3D_degree2_kernels_list = [
    generate_1hop_kernel(3, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]
# degree3
functional_groups = hop1_degree3_functional_groups
hop1_3D_degree3_kernels_list = [
    generate_1hop_kernel(3, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]
# degree4
functional_groups = hop1_degree4_functional_groups
hop1_3D_degree4_kernels_list = [
    generate_1hop_kernel(3, functional_groups[name][0],
                         functional_groups[name][1]) for name in
    functional_groups.keys()]


def get_hop1_kernel_list(D):
    if D == 2:
        # to make sure the original list is untouched, use the copied one
        return hop1_2D_degree1_kernels_list.copy(), hop1_2D_degree2_kernels_list.copy(), hop1_2D_degree3_kernels_list.copy(), hop1_2D_degree4_kernels_list.copy()
    elif D == 3:
        return hop1_3D_degree1_kernels_list.copy(), hop1_3D_degree2_kernels_list.copy(), hop1_3D_degree3_kernels_list.copy(), hop1_3D_degree4_kernels_list.copy()
    else:
        raise Exception('get_hop1_kernel_list(): invalid D')


if __name__ == '__main__':
    list_2D = get_hop1_kernel_list(2)
    list_3D = get_hop1_kernel_list(3)
    for lst in list_2D:
        for item in lst:
            print(item)
    for lst in list_3D:
        for item in lst:
            print(item)

    print_kernel_files()

    # read_kernel_from_csv('customized_kernels/customized_kernel1.csv')
