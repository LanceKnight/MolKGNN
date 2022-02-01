import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import degree
from tqdm import tqdm
import numpy as np

pattern_dict = {'[NH-]': '[N-]'}
add_atom_num = 5
num_reference = 10000  # number of reference molecules for augmentation


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


def get_atom_rep(atomic_num):
    '''use rdkit to generate atom representation
    '''
    global elem_lst

    result = 0
    try:
        result = elem_lst[atomic_num - 1]
    except:
        print(f'error: atomic_num {atomic_num} does not exist')

    return result


def smiles2graph(D, smiles):
    if D == None:
        raise Exception(
            'smiles2grpah() needs to input D to specifiy 2D or 3D graph '
            'generation.')
    # print(f'smiles:{smiles}')
    # Default RDKit behavior is to reject hypervalent P, so you need to set
    # sanitize=False. Search keyword = 'Explicit Valence Error - Partial
    # Sanitization' on https://www.rdkit.org/docs/Cookbook.html for more info
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
    try:
        # mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f'{e}, smiles:{smiles}')

    if D == 2:
        Chem.rdDepictor.Compute2DCoords(mol)
    if D == 3:
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

    atom_pos = []
    atom_attr = []

    # get atom attributes and positions
    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        h = get_atom_rep(atomic_num)

        if D == 2:
            atom_pos.append(
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y])
        elif D == 3:
            atom_pos.append([conf.GetAtomPosition(
                i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
        atom_attr.append(h)

    # get bond attributes
    edge_list = []
    edge_attr_list = []
    for idx, edge in enumerate(mol.GetBonds()):
        i = edge.GetBeginAtomIdx()
        j = edge.GetEndAtomIdx()

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

        edge_list.append((i, j))
        edge_attr_list.append(bond_attr)
        #         print(f'i:{i} j:{j} bond_attr:{bond_attr}')

        edge_list.append((j, i))
        edge_attr_list.append(bond_attr)
    #         print(f'j:{j} j:{i} bond_attr:{bond_attr}')

    x = torch.tensor(atom_attr, dtype=torch.double)
    p = torch.tensor(atom_pos, dtype=torch.double)
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dytpe=torch.double)

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
                edge_attr=edge_attr)  # , adj=adj, attn_bias=attn_bias,
    # attn_edge_type=attn_edge_type)
    # data = preprocess_item(data)
    return data


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
        second one being 'labels'
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
        smiles_list = list(data_df['smiles'])[0:12]
        labels_list = list(data_df['labels'])[0:12]

        for i, smi in tqdm(enumerate(smiles_list)):
            label = labels_list[i]
            data = smiles2graph(self.D, smi)
            if data is None:
                continue
            data.idx = i
            data.y = torch.tensor([label], dtype=torch.int)
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

    def get_idx_split(self):
        indices = np.load(self.idx_file, allow_pickle=True)
        train_indices = indices[0]
        val_indices = indices[1]
        test_indices = indices[2]

        split_dict = {}
        # Delete if statement if using the full CHIRAL1 dataset
        split_dict['train'] = [torch.tensor(x) for x in train_indices if x <
                               1000]
        split_dict['valid'] = [torch.tensor(x) for x in val_indices if
                               x < 1000]
        split_dict['test'] = [torch.tensor(x) for x in test_indices if
                              x < 1000]

        return split_dict


class QSARDataset(InMemoryDataset):
    """
    Dataset from Mariusz Butkiewics et al., 2013, Benchmarking ligand-based
    virtual High_Throughput Screening with the PubChem Database

    There are nine subsets in this dataset, identified by their summary assay
    IDs (SAIDs):
    435008, 1798, 345034, 1843, 2258, 463087, 488997,2689, 485290
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
                 empty=False):

        self.dataset = dataset
        self.root = root
        self.D = D
        super(QSARDataset, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, \
                                                              pre_transform, \
                                                              pre_filter

        if not empty:
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

    @property
    def processed_file_names(self):
        return f'{self.dataset}-{self.D}D.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    # def __getitem__(self, index):
    #     return self.get(index)

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset not in ['435008', '1798', '435034']:
            # print(f'dataset:{self.dataset}')
            raise ValueError('Invalid dataset name')

        for file, label in [(f'{self.dataset}_actives.smi', 1),
                            (f'{self.dataset}_inactives.smi', 0)]:
            smiles_path = os.path.join(self.root, 'raw', file)
            smiles_list = pd.read_csv(
                smiles_path, sep='\t', header=None)[0]

            # Only get first N data, just for debugging
            smiles_list = smiles_list[0:4000]

            for i in tqdm(range(len(smiles_list)), desc=f'{file}'):
                # for i in tqdm(range(1)):
                smi = smiles_list[i]

                data = smiles2graph(self.D, smi)
                if data is None:
                    continue

                # # If use ogb_smiles2graph()
                # try:
                #     graph = ogb_smiles2graph(smi)
                # except:
                #     print('cannot convert smiles to graph')
                #     pass

                # data = Data()
                # data.__num_nodes__ = int(graph['num_nodes'])
                # data.edge_index = torch.from_numpy(graph['edge_index']).to(
                #     torch.int64)
                # data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                #     torch.int64)
                # data.x = torch.from_numpy(graph['node_feat']).to(
                # torch.float32)

                data.idx = i
                data.y = torch.tensor([label], dtype=torch.int)
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
            self.processed_dir, f'{self.dataset}-smiles.csv'), index=False,
            header=False)

        # print(f'data length:{len(data_list)}')
        # for data in data_list:
        #     print(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = {}
        # # Total 362 actives. Split: train-290, 36, 36
        # split_dict['train'] = [torch.tensor(x) for x in range(0, 326)] + [
        #     torch.tensor(x) for x in range(1000, 10674)]  # 10K Training
        # split_dict['valid'] = [torch.tensor(x) for x in range(326, 362)] + [
        #     torch.tensor(x) for x in range(20000, 29964)]  # 10K val
        # split_dict['test'] = [torch.tensor(x) for x in range(326, 362)] + [
        #     torch.tensor(x) for x in range(3000, 9066)]

        # Super small dataset for processing debugging.
        # Total 362 actives. Split: 290, 36, 36
        split_dict['train'] = [torch.tensor(x) for x in range(0, 326)] + [
            torch.tensor(x) for x in range(400, 1074)]  # 1K Training
        split_dict['valid'] = [torch.tensor(x) for x in range(326, 362)] + [
            torch.tensor(x) for x in range(1100, 2064)]  # 1K val
        split_dict['test'] = [torch.tensor(x) for x in range(326, 362)] + [
            torch.tensor(x) for x in range(3000, 4000)]

        return split_dict

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return item
        else:
            return self.index_select(idx)


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
    pass
    # dataset = MyQSARDataset(root='../../dataset/connect_aug/',
    #                            generate_num=5)
