from models.ChIRoNet.embedding_functions import embedConformerWithAllPaths
import math
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.collate import collate
from torch_geometric.utils import degree
from tqdm import tqdm
import numpy as np
import rdkit
import rdkit.Chem.EState as EState
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.rdPartialCharges as rdPartialCharges


pattern_dict = {'[NH-]': '[N-]', '[OH2+]':'[O]'}

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


def one_hot_vector(val, lst):
	'''
	Converts a value to a one-hot vector based on options in lst
	'''
	if val not in lst:
		val = lst[-1]
	return map(lambda x: x == val, lst)


def get_atom_rep(atom):
    features = []
    # H, C, N, O, F, Si, P, S, Cl, Br, I, other
    features += one_hot_vector(atom.GetAtomicNum(), [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 999])
    features += one_hot_vector(len(atom.GetNeighbors()), list(range(1, 5)))

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

    # Get bond attributes
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

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)

    x = torch.tensor(all_atom_features, dtype=torch.float32)
    p = torch.tensor(atom_pos, dtype=torch.float32)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    atomic_num = torch.tensor(atomic_num_list, dtype=torch.int)


    data = Data(x=x, p=p, edge_index=edge_index,
                edge_attr=edge_attr, atomic_num=atomic_num)  # , adj=adj,
    return data

def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph '
            'generation.')
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
        return f'shrink_{self.subset_name}.pt'

    def process(self):
        data_smiles_list = []
        data_list = []
        data_df = pd.read_csv(self.data_file)

        smiles_list = list(data_df['smiles'])
        labels_list = list(data_df[self.label_column_name])


        for i, smi in tqdm(enumerate(smiles_list)):
            label = labels_list[i]
            data = smiles2graph(self.D, smi)
            if data is None:
                continue
            data.idx = i
            data.y = torch.tensor([label], dtype=torch.float)
            data.smiles = smi

            data_list.append(data)
            data_smiles_list.append(smiles_list[i])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            print('doing pre_transforming...')
            data_list = [self.pre_transform(data) for data in data_list]

        # Write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(
            self.processed_dir, f'{self.subset_name}-smiles.csv'), index=False,
            header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        indices = np.load(self.idx_file, allow_pickle=True)
        train_indices = indices[0]
        val_indices = indices[1]
        test_indices = indices[2]

        split_dict = {}
        split_dict['train'] = train_indices
        split_dict['valid'] = val_indices
        split_dict['test'] = test_indices
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
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='435008',
                 empty=False,
                 gnn_type='kgnn'):

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
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return f'{self.gnn_type}-{self.dataset}-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        print(f'processing dataset {self.dataset}')
        if self.dataset not in ['435008', '1798', '435034', '1843', '2258',
                                '463087', '488997','2689', '485290','9999']:
            raise ValueError('Invalid dataset name')

        RDLogger.DisableLog('rdApp.*')

        data_smiles_list = []
        data_list = []
        counter = -1
        invalid_id_list = []
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

                if data is None:
                    invalid_id_list.append([counter, label])
                    continue
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

        # Write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, f'{self.gnn_type}-{self.dataset}-smiles.csv'),
                                  index=False, header=False)
        invalid_id_series = pd.DataFrame(invalid_id_list)
        invalid_id_series.to_csv(os.path.join(self.processed_dir, f'{self.gnn_type}-{self.dataset}-invalid_id.csv'),
                                 index=False,
                                              header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



    def dimenetpp_process(self, mol):
        conformer = mol.GetConformer()
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype=int))  # keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype=int)  # indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2 * array_adj.shape[1]), dtype=int)  # placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)

        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        node_features = np.array(
            [atom.GetAtomicNum() for atom in atoms])  # Z
        positions = np.array(
            [conformer.GetAtomPosition(atom.GetIdx()) for atom in atoms])  # xyz positions
        edge_index, Z, pos = edge_index, node_features, positions
        data = Data(
            x=torch.as_tensor(Z).unsqueeze(1),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long))
        data.pos = torch.as_tensor(pos, dtype=torch.float)
        return data

    def chiro_process(self, mol):

        return_values = embedConformerWithAllPaths(mol, repeats=False)
        if return_values is not None:
            atom_symbols, edge_index, edge_features, node_features, \
            bond_distances, bond_distance_index, bond_angles, \
            bond_angle_index, dihedral_angles, dihedral_angle_index = return_values
        else:
            return

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


    def get_idx_split(self):
        split_dict = torch.load(f'data_split/shrink_{self.dataset}_seed2.pt')
        try:
            invalid_id_list = pd.read_csv(os.path.join(self.processed_dir, f'{self.gnn_type}-'
                                                                       f'{self.dataset}-invalid_id.csv')
                                      , header=None).values.tolist()
            for id, label in invalid_id_list:
                print(f'checking invalid id {id}')
                if label == 1:
                    print('====warning: a positive label is removed====')
                if id in split_dict['train']:
                    split_dict['train'].remove(id)
                    print(f'found in train and removed')
                if id in split_dict['valid']:
                    split_dict['valid'].remove(id)
                    print(f'found in valid and removed')
                if id in split_dict['test']:
                    split_dict['test'].remove(id)
                    print(f'found in test and removed')
        except:
            print(f'invalid_id_list is empty')

        return split_dict

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            # item.idx = idx
            return item
        else:
            return self.index_select(idx)

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


class ToXAndPAndEdgeAttrForDeg(object):
    '''
    Calculate the focal index and neighbor indices for each degree and store
    them in focal_index and nei_index.
    Also calculate neighboring edge attr for each degree nei_edge_attr.
    These operations are very expensive to run on the fly
    '''

    def get_neighbor_index(self, edge_index, center_index):
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = torch.nonzero(b)
        d = c[:, 0]
        return edge_index[1, d]

    def get_degree_index(self, x, edge_index):
        deg = degree(edge_index[0], x.shape[0])
        return deg

    def get_edge_attr_support_from_center_node(self, edge_attr, edge_index,
                                               center_index):
        a = edge_index[0]
        b = a.unsqueeze(1) == center_index
        c = torch.nonzero(b)
        d = c[:, 0]

        # Normalize bond id
        e = (d / 2).long()
        bond_id = torch.tensor([2 * x for x in e], device=a.device)

        # Select bond attributes with the bond id
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

        num_focal = len(focal_index)
        nei_index_list_each_node = []

        nei_p_list_each_node = []
        nei_edge_attr_list_each_node = []

        for i in range(num_focal):
            nei_index = self.get_neighbor_index(edge_index, focal_index[i])
            nei_index_list_each_node.append(nei_index)

            nei_p = torch.index_select(p, 0, nei_index)
            nei_edge_attr = self.get_edge_attr_support_from_center_node(
                edge_attr, edge_index, focal_index[i])

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
    # gnn_type = 'spherenet'
    # gnn_type = 'schnet'
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
    print(f'===={gnn_type}====')

    if gnn_type== 'kgnn':
        transform = ToXAndPAndEdgeAttrForDeg()
    else:
        transform = None

    qsar_dataset = QSARDataset(root='../dataset/qsar/clean_sdf',
                               dataset=args.dataset,
                               pre_transform=transform,
                               gnn_type=args.gnn_type
                               )


    data = qsar_dataset[0]
    print(f'data:{data}')
    print('\n')
    import sys
    print(f'mem size:{sys.getsizeof(data) } bytes')
    print(f'totl mem size = mem_size * 200k /1000 = '
          f'{sys.getsizeof(data) * 200000/1000} MB')
