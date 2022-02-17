from collator import collator
from wrapper import QSARDataset, D4DCHPDataset, ToXAndPAndEdgeAttrForDeg

from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from torchvision.transforms import Lambda
from functools import partial

aug_num = 5
dataset = None


# class ConvertYFromIntToFloat(object):
#     """
#     Convert the y label from integer to float type because loss function
#     requires a float input
#     :return:
#     """
#     def __init__(self):
#         pass
#
#     def __call__(self, data):
#         print(f'before convert:{data.y.dtype}')
#         data.y = data.y.float()
#         print(f'after convert:{data.y.dtype}')
#         return data

def get_dataset(dataset_name='435034'):
    """
    Get the requested dataset
    :param dataset_name:
    :return:
    """
    if dataset_name in ['435008', '1798', '435034']:
        qsar_dataset = QSARDataset(root='../dataset/qsar',
                                   dataset=dataset_name,
                                   pre_transform=ToXAndPAndEdgeAttrForDeg(),
                                   )
        dataset = {
            'num_class': 1,
            'dataset': qsar_dataset,
            'num_samples': len(qsar_dataset),
        }
    elif dataset_name in ['CHIRAL1', 'DIFF5', "dummy"]:
        if dataset_name == 'CHIRAL1':
            data_file =  '../dataset/d4_docking/d4_docking_rs.csv'
            index_file = '../dataset/d4_docking/rs/split0.npy'
        elif dataset_name == 'dummy':
            data_file = '../dataset/d4_docking/dummy/dummy.csv'
            index_file = '../dataset/d4_docking/dummy/split.npy'


        d4_dchp_dataset = D4DCHPDataset(
            root='../dataset/d4_docking/',
            subset_name=dataset_name,
            data_file= data_file,
            idx_file=index_file,
            D=3,
            pre_transform=ToXAndPAndEdgeAttrForDeg(),
        )
        print(f'd4_dchp_dataset:{d4_dchp_dataset}')
        dataset = {
            'num_class': 1,
            'dataset': d4_dchp_dataset,
            'num_samples': len(d4_dchp_dataset)
        }

    else:
        print(f'dataset_name:{dataset_name}')
        raise NotImplementedError(f'data.py::get_dataset: dataset_name is '
                                  f'not found')

    print(f'dataset {dataset_name} loaded!')
    print(dataset)
    print(f'dataset info ends')
    return dataset


class DataLoaderModule(LightningDataModule):
    """
    A pytorch lighning wrapper that creates DataLoaders

    If enable oversampling with replacement, the weights are larger if the
    number of samples is smaller (the probability of drawing a sample is the
    inverse of the number of this sample class
    """

    def __init__(
            self,
            dataset_name,
            num_workers,
            batch_size,
            seed,
            enable_oversampling_with_replacement=False,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.enable_oversampling_with_replacement = \
            enable_oversampling_with_replacement
        self.dataset_train = ...
        self.dataset_val = ...

    def setup(self, stage: str = None):
        split_idx = self.dataset['dataset'].get_idx_split()
        # print(f'split_idx:{split_idx}')

        self.dataset_train = self.dataset['dataset'][split_idx["train"]]
        print(f'training len:{len(self.dataset_train)})')

        self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
        print(f'validation len:{len(self.dataset_val)})')

        self.dataset_test = self.dataset['dataset'][split_idx["test"]]
        print(f'testing len:{len(self.dataset_test)})')

    def train_dataloader(self):
        # Calculate the number of samples in minority/majority class
        num_train_active = len(torch.nonzero(
            torch.tensor([data.y for data in self.dataset_train])))
        num_train_inactive = len(self.dataset_train) - num_train_active
        print(
            f'training size: {len(self.dataset_train)}, actives: '
            f'{num_train_active}')

        if self.enable_oversampling_with_replacement:
            print('data.py::with resampling')
            # Sample weights equal the inverse of number of samples
            train_sampler_weight = torch.tensor([(1. / num_train_inactive)
                                                 if data.y == 0
                                                 else (1. / num_train_active)
                                                 for data in
                                                 self.dataset_train])

            generator = torch.Generator()
            generator.manual_seed(self.seed)

            train_sampler = WeightedRandomSampler(weights=train_sampler_weight,
                                                  num_samples=len(
                                                      train_sampler_weight),
                                                  generator=generator)

            train_loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.num_workers,
            )
        else:  # Regular sampling without oversampling
            print('data.py::no resampling')
            print(f'dataset_train:{self.dataset_train[0]}')
            train_loader = DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

        print('len(train_dataloader)', len(train_loader))

        return train_loader

    def val_dataloader(self):
        print(f'dataset_train:{self.dataset_train[0]}')
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(f'dataset_train:{self.dataset_train[0]}')
        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_loader, train_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator,
                               max_node=get_dataset(self.dataset_name)[
                                   'max_node'],
                               multi_hop_max_dist=self.multi_hop_max_dist,
                               spatial_pos_max=self.spatial_pos_max),
        )
        print('len(test_dataloader)', len(test_loader))
        for batch in test_loader:
            print(batch.y)
        return test_loader

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataLoader")
        parser.add_argument('--dataset_name', type=str, default="435034")
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        return parent_parser
#
# class AugmentedDataModule(LightningDataModule):
#     def __init__(self,
#                  dataset_name: str = 'qsar',
#                  num_workers: int = 0,
#                  batch_size: int = 256,
#                  seed: int = 42,
#                  multi_hop_max_dist: int = 5,
#                  spatial_pos_max: int = 1024,
#                  generate_num=aug_num,
#                  *args,
#                  **kwargs,
#                  ):
#         super(AugmentedDataModule, self).__init__()
#         self.batch_size = batch_size
#         self.generate_num = generate_num
#         self.metric = 'loss'
#         self.multi_hop_max_dist = multi_hop_max_dist
#         self.spatial_pos_max = spatial_pos_max
#
#     def setup(self, stage=None):
#         self.train_set = AugmentedDataset(root='../../dataset/connect_aug',
#                                           generate_num=self.generate_num)
#         self.val_set = self.train_set[:2]
#         print(
#             f'len(trainset):{len(self.train_set)} len(val_set):'
#             f'{len(self.val_set)}')
#
#     def train_dataloader(self):
#         print('train loader here')
#         loader = DataLoader(self.train_set, batch_size=self.batch_size,
#                             shuffle=True,
#                             collate_fn=partial(collator, max_node=38,
#                                                multi_hop_max_dist=self.multi_hop_max_dist,
#                                                spatial_pos_max=self.spatial_pos_max), )
#         # for batch in loader:
#         #     print(batch)
#         return loader
#
#     def val_dataloader(self):
#         print('pretrain validation loader here')
#         val_loader = DataLoader(self.val_set, batch_size=self.batch_size,
#                                 collate_fn=partial(collator, max_node=38,
#                                                    multi_hop_max_dist=self.multi_hop_max_dist,
#                                                    spatial_pos_max=self.spatial_pos_max), )
#         return val_loader
#
#         # # test set
#         # smi = 'C1(=CC=CC(=C1)C(CC)C)O'
#         # data1 = smiles2graph(2, smi)
#         # smi = 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(
#         # =N4)C5=CN=CC=C5'
#         # data2 = smiles2graph(2, smi)
#         # smi = 'C1=CC=CC(=C1)C(CC)C'
#         # data3 = smiles2graph(2, smi)
#         # dataset = [data1, data2, data3]
#         #
#         # self.test_set = dataset
