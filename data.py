from wrapper import QSARDataset, D4DCHPDataset, ToXAndPAndEdgeAttrForDeg
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader


qsar_dataset_names= ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290', '9999']
d4dchp_dataset_names = ['CHIRAL1', 'DIFF5', 'D4DCHP', "dummy"]

def get_dataset(dataset_name='435034', gnn_type='kgnn',
                dataset_path='../dataset/'):
    """
    Get the requested dataset
    :param dataset_name:
    :return:
    """
    if gnn_type == 'kgnn':
        pre_transform=ToXAndPAndEdgeAttrForDeg()
    else:
        pre_transform=None

    if dataset_name in qsar_dataset_names:
        qsar_dataset = QSARDataset(
            root=dataset_path+'qsar/clean_sdf',
            dataset=dataset_name,
            gnn_type=gnn_type,
            pre_transform=pre_transform,
            )

        dataset = {
            'num_class': 1,
            'dataset': qsar_dataset,
            'num_samples': len(qsar_dataset),
            'metrics': ['ppv', 'logAUC_0.001_0.1', 'logAUC_0.001_1', 'f1_score', 'AUC'],
            'loss_func': BCEWithLogitsLoss()
        }


    elif dataset_name in d4dchp_dataset_names:
        if dataset_name == 'CHIRAL1':
            data_file =  '../dataset/d4_docking/d4_docking_rs.csv'
            label_column_name = 'labels'
            index_file = '../dataset/d4_docking/rs/split0.npy'
            metrics = ['accuracy']
            loss_func = BCEWithLogitsLoss()
        elif dataset_name == 'D4DCHP':
            data_file = '../dataset/d4_docking/d4_docking.csv'
            label_column_name = 'docking_score'
            index_file = '../dataset/d4_docking/full/split0.npy'
            metrics = ['RMSE']
            loss_func = MSELoss(reduction='sum')
        elif dataset_name == 'dummy':
            data_file = '../dataset/d4_docking/dummy/dummy.csv'
            label_column_name = 'labels'
            index_file = '../dataset/d4_docking/dummy/split.npy'
            metrics = ['accuracy']
            loss_func = BCEWithLogitsLoss()


        d4_dchp_dataset = D4DCHPDataset(
            root='../dataset/d4_docking/',
            subset_name=dataset_name,
            data_file= data_file,
            label_column_name=label_column_name,
            idx_file=index_file,
            D=3,
            pre_transform=ToXAndPAndEdgeAttrForDeg(),
        )

        dataset = {
            'num_class': 1,
            'dataset': d4_dchp_dataset,
            'num_samples': len(d4_dchp_dataset),
            'metrics':metrics,
            'loss_func': loss_func
        }

    else:
        raise NotImplementedError(f'data.py::get_dataset: dataset_name '
                                  f'{dataset_name} is '
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
            enable_oversampling_with_replacement,
            gnn_type,
            dataset_path
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset = get_dataset(dataset_name=self.dataset_name,
                                   gnn_type=gnn_type,
                                   dataset_path = dataset_path
                                   )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.enable_oversampling_with_replacement = enable_oversampling_with_replacement
        self.gnn_type = gnn_type
        self.dataset_path = dataset_path
        split_idx = self.dataset['dataset'].get_idx_split(seed=self.seed)

        self.dataset_train = self.dataset['dataset'][split_idx["train"]]
        print(f'training # samples:{len(self.dataset_train)})')

        self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
        print(f'validation # samples:{len(self.dataset_val)})')

        self.dataset_test = self.dataset['dataset'][split_idx["test"]]
        print(f'testing # samples:{len(self.dataset_test)})')

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        if self.dataset_name in qsar_dataset_names:
            # Calculate the number of samples in minority/majority class
            num_train_active = len(torch.nonzero(
                torch.tensor([data.y for data in self.dataset_train])))
            num_train_inactive = len(self.dataset_train) - num_train_active
            print(f'training # of molecules: {len(self.dataset_train)}, actives: {num_train_active}')

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
                train_loader = DataLoader(
                    self.dataset_train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )

            print('len(train_dataloader)', len(train_loader))
        elif self.dataset_name in d4dchp_dataset_names:
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
        # Validation laader
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        # Train loader in evaluation mode
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return val_loader, train_loader

    def test_dataloader(self):
        # Test laader

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return test_loader

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataLoader")
        parser.add_argument('--dataset_name', type=str, default="435034")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=17)
        parser.add_argument('--enable_oversampling_with_replacement', action='store_true', default=False)
        parser.add_argument('--dataset_path', type=str, default="../dataset/")
        return parent_parser

