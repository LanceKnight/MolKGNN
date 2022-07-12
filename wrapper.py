from models.ChIRoNet.embedding_functions import embedConformerWithAllPaths

import copy
import math
import os
import os.path as osp
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.data.in_memory_dataset import nested_iter
from torch_geometric.data.separate import separate
from torch_geometric.data.collate import collate
from torch_geometric.utils import degree
from tqdm import tqdm
import numpy as np
import random
import rdkit
import rdkit.Chem.EState as EState
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.rdPartialCharges as rdPartialCharges


pattern_dict = {'[NH-]': '[N-]', '[OH2+]':'[O]'}
add_atom_num = 5
num_reference = 10000  # number of reference molecules for augmentation
num_data = 1000 # number of data used for debugging. Not used in full dataset

def smiles_cleaner(smiles):
    '''
    This function is to clean smiles for some known issues that makes
    rdkit:Chem.MolFromSmiles not working
    '''
    print('fixing smiles for rdkit...')
    new_smiles = smiles
    for pattern, replace_value in pattern_dict.items():
        if pattern in smiles:
            print('found pattern and fixed the smiles!')
            new_smiles = smiles.replace(pattern, replace_value)
    return new_smiles


def generate_element_rep_list(elements):
    '''
    Generate an element representation.
    contains weight, van der waal radius, valence, out_eletron
    :param elements:
    :return:
    '''
    print('calculating rdkit element representation lookup table')
    elem_rep_lookup = []
    for elem in elements:
        pt = Chem.GetPeriodicTable()

        if isinstance(elem, int):
            num = elem
            sym = pt.GetElementSymbol(num)
        else:
            num = pt.GetAtomicNumber(elem)
            sym = elem
        w = pt.GetAtomicWeight(elem)

        Rvdw = pt.GetRvdw(elem)
        #     Rcoval = pt.GetRCovalent(elem)
        valence = pt.GetDefaultValence(elem)
        outer_elec = pt.GetNOuterElecs(elem)

        elem_rep = [num, w, Rvdw, valence, outer_elec]
        #             print(elem_rep)

        elem_rep_lookup.append(elem_rep)
    elem_lst = elem_rep_lookup.copy()
    return elem_rep_lookup


max_elem_num = 118
element_nums = [x + 1 for x in range(max_elem_num)]
elem_lst = generate_element_rep_list(element_nums)

def one_hot_vector(val, lst):
	'''
	Converts a value to a one-hot vector based on options in lst
	'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)
# def get_element_rep(atomic_num):
#     '''use rdkit to generate atom representation
#     '''
#     global elem_lst
#
#     result = 0
#     try:
#         result = elem_lst[atomic_num - 1]
#     except:
#         print(f'error: atomic_num {atomic_num} does not exist')
#
#     return result

def get_atom_rep(atom):
    features = []
    features += one_hot_vector(atom.GetAtomicNum(), [6, 7, 8, 9, 15, 16, 17, 35, 53, 999])  # list(range(1, 53))))
    features += one_hot_vector(len(atom.GetNeighbors()), list(range(0, 5)))
    # features.append(atom.GetTotalNumHs())
    features.append(atom.GetFormalCharge())
    features.append(atom.IsInRing())
    features.append(atom.GetIsAromatic())
    features.append(atom.GetExplicitValence())
    features.append(atom.GetMass())

    # Add Gasteiger charge and set to 0 if it is NaN or Infinite
    gasteiger_charge = float(atom.GetProp('_GasteigerCharge'))
    if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
        gasteiger_charge = 0
    features.append(gasteiger_charge)

    # Add Gasteiger H charge and set to 0 if it is NaN or Infinite
    gasteiger_h_charge = float(atom.GetProp('_GasteigerHCharge'))
    if math.isnan(gasteiger_h_charge) or math.isinf(gasteiger_h_charge):
        gasteiger_h_charge = 0

    features.append(gasteiger_h_charge)
    return features

def get_extra_atom_feature(all_atom_features, mol):
    '''
    Get more atom features that cannot be calculated only with atom,
    but also with mol
    :param all_atom_features:
    :param mol:
    :return:
    '''
    # Crippen has two parts: first is logP, second is Molar Refactivity(MR)
    all_atom_crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    all_atom_TPSA_contrib = rdMolDescriptors._CalcTPSAContribs(mol)
    # ASA = Accessible Surface Area
    all_atom_ASA_contrib = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    all_atom_EState = EState.EStateIndices(mol)

    new_all_atom_features = []
    for atom_id, feature in enumerate(all_atom_features):
        crippen_logP = all_atom_crippen[atom_id][0]
        crippen_MR = all_atom_crippen[atom_id][1]
        atom_TPSA_contrib = all_atom_TPSA_contrib[atom_id]
        atom_ASA_contrib = all_atom_ASA_contrib[atom_id]
        atom_EState = all_atom_EState[atom_id]

        feature.append(crippen_logP)
        feature.append(crippen_MR)
        feature.append(atom_TPSA_contrib)
        feature.append(atom_ASA_contrib)
        feature.append(atom_EState)

        new_all_atom_features.append(feature)
    return new_all_atom_features


def mol2graph(mol, D=3):
    try:
        conf = mol.GetConformer()
    except Exception as e:
        smiles = AllChem.MolToSmiles(mol)
        print(f'smiles:{smiles} error message:{e}')

    atom_pos = []
    atomic_num_list = []
    all_atom_features = []

    # Get atom attributes and positions
    rdPartialCharges.ComputeGasteigerCharges(mol)

    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        atomic_num_list.append(atomic_num)
        atom_feature = get_atom_rep(atom)
        # h = get_atom_rep(atomic_num)

        if D == 2:
            atom_pos.append(
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        elif D == 3:
            atom_pos.append([conf.GetAtomPosition(
                i).x, conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z])
        all_atom_features.append(atom_feature)
    # Add extra features that are needs to calculate using mol
    all_atom_features = get_extra_atom_feature(all_atom_features, mol)

    # get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, bond in enumerate(mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_attr = []
        bond_attr += one_hot_vector(
            bond.GetBondTypeAsDouble(),
            [1.0, 1.5, 2.0, 3.0]
        )

        is_aromatic = bond.GetIsAromatic()
        is_conjugate = bond.GetIsConjugated()
        is_in_ring = bond.IsInRing()
        bond_attr.append(is_aromatic)
        bond_attr.append(is_conjugate)
        bond_attr.append(is_in_ring)

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)
        #         print(f'i:{i} j:{j} bond_attr:{bond_attr}')

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)
    #         print(f'j:{j} j:{i} bond_attr:{bond_attr}')

    x = torch.tensor(all_atom_features, dtype=torch.float32)
    p = torch.tensor(atom_pos, dtype=torch.float32)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    atomic_num = torch.tensor(atomic_num_list, dtype=torch.int)

    # graphormer-specific features
    # adj = torch.zeros([N, N], dtype=torch.bool)
    # adj[edge_index[0, :], edge_index[1, :]] = True
    # attn_bias = torch.zeros(
    #     [N + 1, N + 1], dtype=torch.float)  # with graph token
    # attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)],
    # dtype=torch.long)
    # attn_edge_type[edge_index[0, :], edge_index[1, :]
    #                ] = convert_to_single_emb(edge_attr) + 1

    data = Data(x=x, p=p, edge_index=edge_index,
                edge_attr=edge_attr, atomic_num=atomic_num)  # , adj=adj,
    # attn_bias=attn_bias,
    # attn_edge_type=attn_edge_type)
    # data = preprocess_item(data)
    return data

def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph '
            'generation.')
    # print(f'smiles:{smiles}')
    # Default RDKit behavior is to reject hypervalent P, so you need to set
    # sanitize=False. Search keyword = 'Explicit Valence Error - Partial
    # Sanitization' on https://www.rdkit.org/docs/Cookbook.html for more
    # info
    smiles = smiles.replace(r'/=', '=')
    smiles = smiles.replace(r'\=', '=')
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as e:
        print(f'Cannot generate mol, error:{e}, smiles:{smiles}')

    if mol is None:
        smiles = smiles_cleaner(smiles)
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        except Exception as e:
            print(f'Generated mol is None, error:{e}, smiles:{smiles}')
            return None
        if mol is None:
            print(f'Generated mol is still None after cleaning, smiles'
                  f':{smiles}')
    try:
        # mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f'error in adding Hs{e}, smiles:{smiles}')

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f'smiles:{smiles} error message:{e}')

    data = mol2graph(mol)
    return data


def process_smiles(dataset, root, D):
    data_smiles_list = []
    data_list = []
    for file, label in [(f'{dataset}_actives.smi', 1),
                        (f'{dataset}_inactives.smi', 0)]:
        smiles_path = os.path.join(root, 'raw', file)
        smiles_list = pd.read_csv(
            smiles_path, sep='\t', header=None)[0]

        # Only get first N data, just for debugging
        smiles_list = smiles_list

        for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
            # for i in tqdm(range(1)):
            smi = smiles_list[i]

            data = smiles2graph(D, smi)
            if data is None:
                continue

            data.idx = i
            data.y = torch.tensor([label], dtype=torch.int)
            data.smiles = smi

            data_list.append(data)
            data_smiles_list.append(smiles_list[i])
    return data_list, data_smiles_list


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset,
                                  dtype=torch.long)
    x = x + feature_offset
    return x


class D4DCHPDataset(InMemoryDataset):
    """
    Dataset from Langnajit Pattanaik et al., 2020, Message Passing Networks
    for Molecules with Tetrahedral Chirality
    The dataset itself is a subset of the screening result for the protein
    protein-ligand docking for D4 dopamine receptor that only keeps
    stereoisomer pairs for a single 1,3-dicyclohexylpropane skeletal scaffold.
    There are totally 287,468 molecules in D4DCHP, with two subsets in this
    dataset, DIFF5 and DHIRAL1. DIFF5 has 119,166 molecules and contains
    enantiomers exhibiting docking score >5kcal/mol;CHIRAL1 has 204,778
    molecules and contains molecules having a single tetrahedral center.
    """

    def __init__(self,
                 root,
                 subset_name,
                 data_file,
                 label_column_name,
                 idx_file,
                 D,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        """
        :param subset_name: a string. Values can be "FULL", "CHIRAL4" or
        "DIFF5"
        :param data_file: a file containing SMILES. File format: .csv file
        with headers; two columns with the first header being 'smiles' and the
        second one having a column name specifed by param label_column_name
        :param label_column_name: a string of the column name for the label.
        e.g., "docking_score"
        :param split_idx: a file specifying the split indices of samples in
        data_file. File format: a .npy file that should be loaded with
        numpy.load('split_idx.npy', allow_pickle=True). After loading,
        it's a list of 3 items. Training indices are stored as 0th item,
        and validation, test are stored as 1st and 2nd items respectively.
        :param D: a integer being either 2 or 3, meaning the dimension
        """
        self.root = root
        print(f'root:{root}')
        self.subset_name = subset_name
        self.data_file = data_file
        self.label_column_name = label_column_name
        self.idx_file = idx_file
        self.D = D
        super(D4DCHPDataset, self).__init__(root, transform, pre_transform,
                                            pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return f'{self.subset_name}.pt'

    def process(self):
        data_smiles_list = []
        data_list = []
        data_df = pd.read_csv(self.data_file)

        smiles_list = list(data_df['smiles'])#[0:num_data]
        labels_list = list(data_df[self.label_column_name])#[0:num_data]


        for i, smi in tqdm(enumerate(smiles_list)):
            label = labels_list[i]
            data = smiles2graph(self.D, smi)
            if data is None:
                continue
            data.idx = i
            data.y = torch.tensor([label], dtype=torch.float)
            # data.dummy_graph_embedding = torch.ones(1, 32)
            data.smiles = smi

            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print('doing pre_transforming...')
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(
            self.processed_dir, f'{self.subset_name}-smiles.csv'), index=False,
            header=False)

        # print(f'data length:{len(data_list)}')
        # for data in data_list:
        #     print(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, seed):
        indices = np.load(self.idx_file, allow_pickle=True)
        train_indices = indices[0]
        val_indices = indices[1]
        test_indices = indices[2]

        split_dict = {}
        split_dict['train'] = train_indices
        split_dict['valid'] = val_indices
        split_dict['test'] = test_indices

        # # Delete if statement if using the full CHIRAL1 dataset
        # split_dict['train'] = [torch.tensor(x) for x in train_indices if x <  num_data]
        # split_dict['valid'] = [torch.tensor(x) for x in val_indices if x < num_data]
        # split_dict['test'] = [torch.tensor(x) for x in test_indices if x < num_data]

        return split_dict


class QSARDataset(InMemoryDataset):
    """
    Dataset from Mariusz Butkiewics et al., 2013, Benchmarking ligand-based
    virtual High_Throughput Screening with the PubChem Database

    There are nine subsets in this dataset, identified by their summary assay
    IDs (SAIDs):
    435008, 1798, 435034, 1843, 2258, 463087, 488997,2689, 485290
    The statistics of each subset can be found in the original publication
    """

    def __init__(self,
                 root,
                 D=3,
                 # data = None,
                 # slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='435008',
                 empty=False,
                 gnn_type='kgnn',
                 seed = 42):

        self.dataset = dataset
        self.root = root
        self.D = D
        self.gnn_type = gnn_type
        super(QSARDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, \
                                                              pre_transform, \
                                                              pre_filter

        if not empty:
            # self.data = torch.load(self.processed_paths[0]) # For non-InMemoryDataset
            self.data, self.slices = torch.load(self.processed_paths[0])

    # def get(self, idx):
    #     data = Data()
    #     for key in self.data.keys:
    #         item, slices = self.data[key], self.slices[key]
    #         s = list(repeat(slice(None), item.dim()))
    #         s[data.__cat_dim__(key, item)] = slice(slices[idx],
    #                                                 slices[idx + 1])
    #         data[key] = item[s]
    #     return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    # @property
    # def processed_dir(self):
    #     folder_name = ''
    #     if self.gnn_type in ['kgnn']:
    #         folder_name = f'kgnn-based-{self.dataset}-{self.D}D'
    #     elif self.gnn_type in ['chironet']:
    #         folder_name = f'chironet-based-{self.dataset}-{self.D}D'
    #     elif self.gnn_type in ['dimenet_pp', 'schnet',
    #                            'spherenet']:
    #         folder_name = f'dimenetpp-based-{self.dataset}-{self.D}D'
    #     else:
    #         NotImplementedError('wrapper.py gnn_type is not defined for '
    #                             'processed dataset')
    #     return osp.join(self.root, f'processed/{folder_name}')


    @property
    def processed_file_names(self):
        # data_list = []
        # counter=-1
        # for file_name, label in [(f'{self.dataset}_actives_new.sdf', 1),
        #                          (f'{self.dataset}_inactives_new.sdf', 0)]:
        #     sdf_path = os.path.join(self.root, 'raw', file_name)
        #     sdf_supplier = Chem.SDMolSupplier(sdf_path)
        #     for i in range(len(sdf_supplier)):
        #         counter+=1
        #         data_list.append(f'data_{counter}.pt')
        #     if label == 1:
        #         self.num_actives = len(sdf_supplier)
        #     else:
        #         self.num_inactives = len(sdf_supplier)
        # return data_list
        return f'{self.dataset}-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        print(f'processing dataset {self.dataset}')
        if self.dataset not in ['435008', '1798', '435034', '1843', '2258',
                                '463087', '488997','2689', '485290','9999']:
            # print(f'dataset:{self.dataset}')
            raise ValueError('Invalid dataset name')

        RDLogger.DisableLog('rdApp.*')

        data_smiles_list = []
        data_list = []
        counter = -1
        for file_name, label in [(f'{self.dataset}_actives_new.sdf', 1),
                                 (f'{self.dataset}_inactives_new.sdf', 0)]:
            sdf_path = os.path.join(self.root, 'raw', file_name)
            sdf_supplier = Chem.SDMolSupplier(sdf_path)
            for i, mol in tqdm(enumerate(sdf_supplier)):
                counter+=1
                if self.gnn_type == 'chironet':
                    data = self.chiro_process(mol)
                elif self.gnn_type in ['dimenet_pp', 'schnet', 'spherenet']:
                    data = self.dimenetpp_process(mol)
                else:
                    data = self.regular_process(mol)

                data.idx = counter
                data.y = torch.tensor([label], dtype=torch.int)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                smiles = AllChem.MolToSmiles(mol)
                data.smiles = smiles

                data_list.append(data)
                data_smiles_list.append(smiles)

                # # Write to processed file
                # torch.save(data, osp.join(self.processed_dir, f'data_'
                #                                               f'{counter}.pt'))


        # Write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(
            self.processed_dir, f'{self.dataset}-smiles.csv'), index=False,
            header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



    def dimenetpp_process(self, mol):
        conformer = mol.GetConformer()
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj,
                               dtype=int))  # keeping just upper
        # triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj),
                             dtype=int)  # indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2 * array_adj.shape[1]),
                              dtype=int)  # placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)

        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = np.array(
            [atom.GetAtomicNum() for atom in atoms])  # Z
        positions = np.array(
            [conformer.GetAtomPosition(atom.GetIdx()) for atom in
             atoms])  # xyz positions
        edge_index, Z, pos = edge_index, node_features, positions
        data = Data(
            x=torch.as_tensor(Z).unsqueeze(1),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long))
        data.pos = torch.as_tensor(pos, dtype=torch.float)
        return data

    def chiro_process(self, mol):
        atom_symbols, edge_index, edge_features, node_features, \
        bond_distances, bond_distance_index, bond_angles, \
        bond_angle_index, dihedral_angles, dihedral_angle_index = \
            embedConformerWithAllPaths(
            mol, repeats=False)

        bond_angles = bond_angles % (2 * np.pi)
        dihedral_angles = dihedral_angles % (2 * np.pi)

        data = Data(
            x=torch.as_tensor(node_features),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(edge_features))
        data.bond_distances = torch.as_tensor(bond_distances)
        data.bond_distance_index = torch.as_tensor(bond_distance_index,
                                                   dtype=torch.long).T
        data.bond_angles = torch.as_tensor(bond_angles)
        data.bond_angle_index = torch.as_tensor(bond_angle_index,
                                                dtype=torch.long).T
        data.dihedral_angles = torch.as_tensor(dihedral_angles)
        data.dihedral_angle_index = torch.as_tensor(
            dihedral_angle_index, dtype=torch.long).T

        return data

    def regular_process(self, mol):
        data = mol2graph(mol)
        return data


    def get_idx_split(self, seed):
        split_dict = torch.load(f'data_split/shrink_{self.dataset}_seed2.pt')
        return split_dict

    # For non-InMemDataset
    # def get(self, idx):
    #     data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    #     return data

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return item
        else:
            return self.index_select(idx)

    # def len(self):
    #     return len(self.processed_file_names)

    @staticmethod
    def collate(data_list):
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    # def __getitem__(self, idx):
    #     if isinstance(idx, int):
    #         item = self.get(self.indices()[idx])
    #         item.idx = idx
    #         return item
    #     else:
    #         return self.index_select(idx)


class ToXAndPAndEdgeAttrForDeg(object):
    '''
    Calculate the focal index and neighbor indices for each degree and store
    them in focal_index and nei_index.
    Also calculate neighboring edge attr for each degree nei_edge_attr.
    These operations are very expensive to run on the fly
    '''

    def get_neighbor_index(self, edge_index, center_index):
        #         print('edge_index')
        #         print(edge_index)
        #         print('\n')
        #         print('center_index')
        #         print(center_index)
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = torch.nonzero(b)
        d = c[:, 0]
        return edge_index[1, d]

    def get_degree_index(self, x, edge_index):
        # print(f'edge_index:{edge_index.shape}, x:{x.shape}')
        deg = degree(edge_index[0], x.shape[0])
        return deg

    def get_edge_attr_support_from_center_node(self, edge_attr, edge_index,
                                               center_index):
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = torch.nonzero(b)
        d = c[:, 0]

        # normalize bond id
        e = (d / 2).long()
        #         bond_id = torch.cat([torch.stack((2*x, 2*x+1)) for x in e])
        bond_id = torch.tensor([2 * x for x in e], device=a.device)
        #         print('bond_id')
        #         print(bond_id)

        # select bond attributes with the bond id
        nei_edge_attr = torch.index_select(input=edge_attr, dim=0,
                                           index=bond_id)

        return nei_edge_attr

    def convert_grpah_to_receptive_field_for_degN(self, deg, deg_index, data):
        x = data.x
        p = data.p
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        selected_index = focal_index = \
            (deg_index == deg).nonzero(as_tuple=True)[0]

        p_focal = torch.index_select(input=p, dim=0, index=focal_index)

        # Debug
        # if deg == 2:
        #     print(f'convert():p:{p}')
        #     print(f'convert():focal_index:{focal_index}')
        #     print(f"convert():p_focal:{p_focal}")

        num_focal = len(focal_index)
        nei_index_list_each_node = []

        nei_p_list_each_node = []
        nei_edge_attr_list_each_node = []

        for i in range(num_focal):
            nei_index = self.get_neighbor_index(edge_index, focal_index[i])
            nei_index_list_each_node.append(nei_index)

            # nei_x = torch.index_select(x, 0, nei_index)
            # #             print(f'nei_x:{nei_x.shape}')
            nei_p = torch.index_select(p, 0, nei_index)
            #             print(f'nei_p:{nei_p.shape}')
            nei_edge_attr = self.get_edge_attr_support_from_center_node(
                edge_attr, edge_index, focal_index[i])
            #             print('\n nei_edge_attr')
            #             print(nei_edge_attr)

            # nei_x_list_each_node.append(nei_x)
            nei_p_list_each_node.append(nei_p)
            nei_edge_attr_list_each_node.append(nei_edge_attr)

        if num_focal != 0:
            nei_index = torch.stack(nei_index_list_each_node, dim=0).reshape(
                -1)
            nei_p = torch.stack(nei_p_list_each_node, dim=0)
            nei_edge_attr = torch.stack(nei_edge_attr_list_each_node, dim=0)
        else:
            nei_index = torch.Tensor()
            nei_p = torch.Tensor()
            nei_edge_attr = torch.Tensor()

        nei_index = nei_index.to(torch.long)

        return p_focal, nei_p, nei_edge_attr, \
               selected_index, nei_index

    def __call__(self, data):

        deg_index = self.get_degree_index(data.x, data.edge_index)

        data.p_focal_deg1 = data.p_focal_deg2 = data.p_focal_deg3 = \
            data.p_focal_deg4 = None
        data.nei_p_deg1 = data.nei_p_deg2 = data.nei_p_deg3 = \
            data.nei_p_deg4 = None
        data.nei_edge_attr_deg1 = data.nei_edge_attr_deg2 = \
            data.nei_edge_attr_deg3 = data.nei_edge_attr_deg4 = None

        # x_focal_list = [data.x_focal_deg1, data.x_focal_deg2,
        # data.x_focal_deg3, data.x_focal_deg4]
        # p_focal_list = [data.p_focal_deg1, data.p_focal_deg2,
        # data.p_focal_deg3, data.p_focal_deg4]
        # nei_x_list = [data.nei_x_deg1, data.nei_x_deg2, data.nei_x_deg3,
        # data.nei_x_deg4]
        # nei_p_list = [data.nei_p_deg1, data.nei_p_deg2, data.nei_p_deg3,
        # data.nei_p_deg4]
        # nei_edge_attr_list = [data.nei_edge_attr_deg1,
        # data.nei_edge_attr_deg2, data.nei_edge_attr_deg3,
        # data.nei_edge_attr_deg4]

        deg = 1
        data.p_focal_deg1, data.nei_p_deg1, data.nei_edge_attr_deg1, \
        data.selected_index_deg1, data.nei_index_deg1 = \
            self.convert_grpah_to_receptive_field_for_degN(
            deg, deg_index, data)

        deg = 2
        data.p_focal_deg2, data.nei_p_deg2, data.nei_edge_attr_deg2,\
        data.selected_index_deg2, data.nei_index_deg2 = \
            self.convert_grpah_to_receptive_field_for_degN(
            deg, deg_index, data)

        # # Debug
        # print(f'wrapper.py==================')
        # print(f'wrapper.py::smiles:{data.smiles}')
        # print(f'wrapper.py::p_focal:{data.p_focal_deg2}')

        deg = 3
        data.p_focal_deg3, data.nei_p_deg3, data.nei_edge_attr_deg3, \
        data.selected_index_deg3, data.nei_index_deg3 = \
            self.convert_grpah_to_receptive_field_for_degN(
            deg, deg_index, data)

        deg = 4
        data.p_focal_deg4, data.nei_p_deg4, data.nei_edge_attr_deg4, \
        data.selected_index_deg4, data.nei_index_deg4 = \
            self.convert_grpah_to_receptive_field_for_degN(
            deg, deg_index, data)

        return data



if __name__ == "__main__":
    from clearml import Task
    from argparse import ArgumentParser

    gnn_type = 'kgnn'
    # gnn_type = 'chironet'
    # gnn_type = 'dimenet_pp'
    use_clearml = False
    if use_clearml:
        task = Task.init(project_name=f"DatasetCreation/kgnn",
                         task_name=f"{gnn_type}",
                         tags=[],
                         reuse_last_task_id=False
                         )

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='1798')
    parser.add_argument('--gnn_type', type=str, default=gnn_type)
    parser.add_argument('--task_name', type=str, default='Unnamed')
    args = parser.parse_args()
    if use_clearml:
        print(f'change_task_name...')
        task.set_name(args.task_name)

    qsar_dataset = QSARDataset(root='../dataset/qsar/clean_sdf',
                               dataset=args.dataset,
                               pre_transform=ToXAndPAndEdgeAttrForDeg(),
                               gnn_type=args.gnn_type
                               )


    data = qsar_dataset[0]
    print(f'data:{data}')
    print('\n')
    import sys
    print(f'mem size:{sys.getsizeof(data) } bytes')
    print(f'totl mem size = mem_size * 200k /1000 = '
          f'{sys.getsizeof(data) * 200000/1000} MB')