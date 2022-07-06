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


def get_info(dataset):
    filename=f'logs/dataset{dataset}_info.log'
    write_and_print(f'loading dataset {dataset}', file=filename)
    start = time.time()
    qsar_dataset = QSARDataset(root='../dataset/qsar/clean_sdf',
                               dataset=dataset,
                               pre_transform=ToXAndPAndEdgeAttrForDeg(),
                               gnn_type='kgnn'
                               )

    loader = DataLoader(qsar_dataset[:], batch_size=16, num_workers=10)
    # loader = loader.to(device)
    end = time.time()
    write_and_print(f'done. loading_time{end-start}', file=filename)
    tot_nodes = 0
    tot_edges = 0
    tot_graphs = len(qsar_dataset)
    start = time.time()
    for data in tqdm(loader):
        # pass
        tot_nodes+=data.x.shape[0]
        tot_edges+=data.num_edges
    end = time.time()

    write_and_print(f'done. calculation_time{end-start}', file=filename)
    write_and_print(f'tot_graphs:{tot_graphs}', file=filename)
    write_and_print(f'tot_nodes:{tot_nodes}, avg_nodes:{tot_nodes/tot_graphs}', file=filename)
    write_and_print(f'tot_edges:{tot_edges}, avg_edges:{tot_edges/tot_graphs}', file=filename)


if __name__  == '__main__':
    dataset_list = [ '9999','485290', '1843', '2258', '488997','2689', '435008', '435034', '463087', '1798'] # arg0
    dataset_list = ['435008', '435034', '463087', '1798'] # arg0
    dataset_list = ['1798']

    for dataset in dataset_list:
        get_info(dataset)

    # with Pool(processes = 9) as pool: # Change this
    #     pool.map(get_info, dataset_list)

    # get_info('1798')