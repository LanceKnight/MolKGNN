from wrapper import QSARDataset, ToXAndPAndEdgeAttrForDeg
from torch_geometric.loader.dataloader import DataLoader
import time
from tqdm import tqdm
import torch

import multiprocessing as mp
from multiprocessing import Pool, Value





def write_and_print(input_str, file=f'logs/dataset_info.log'):
    print(input_str)
    with open(file, 'a+') as out_file:
        out_file.write(input_str)
        out_file.write('\n')


def get_info(dataset, batch_size=16):
    filename=f'logs/dataset{dataset}_info.log'
    write_and_print(f'loading dataset {dataset}', file=filename)
    start = time.time()
    qsar_dataset = QSARDataset(root='../dataset/qsar/clean_sdf',
                               dataset=dataset,
                               pre_transform=ToXAndPAndEdgeAttrForDeg(),
                               gnn_type='kgnn'
                               )

    loader = DataLoader(qsar_dataset[:], batch_size=batch_size, num_workers=10)
    # loader = loader.to(device)
    end = time.time()
    write_and_print(f'done. loading_time{end-start}', file=filename)
    tot_actives = 0
    tot_inactives = 0
    tot_graphs = len(qsar_dataset)
    start = time.time()
    for data in tqdm(loader):
        # pass
        num_non_zeros = torch.count_nonzero(data.y).item()
        tot_actives+=num_non_zeros
        tot_inactives+=(len(data.y)-num_non_zeros)

    end = time.time()

    write_and_print(f'done. calculation_time{end-start}', file=filename)
    write_and_print(f'tot_graphs:{tot_graphs}, tot_actives:{tot_actives}, tot_inactives:{tot_inactives}', file=filename)



if __name__  == '__main__':
    # dataset_list = [ '9999','485290', '1843', '2258', '488997','2689', '435008', '435034', '463087', '1798'] # arg0
    # dataset_list = ['435008', '435034', '463087', '1798'] # arg0
    dataset_list = ['463087']


    for dataset in dataset_list:
        get_info(dataset, batch_size =16)

