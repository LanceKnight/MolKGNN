import multiprocessing as mp
from multiprocessing import Pool, Value
import os
from tqdm import tqdm
import shutil, errno
import itertools
import time





def run_command(exp_id, dataset): # Change this
    # Model=kgnn
    os.system(f'python wrapper.py \
        --task_name dataset_task{exp_id}\
        --dataset {dataset}\
        ')


def copyanything(src, dst):
    # If dst exits, remove it first
    if os.path.exists(dst):
        shutil.rmtree(dst)
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def run(exp_id, dataset):
    exp_name = f'creating{exp_id}_dataset{dataset}' # Change this
    print(f'=====running {exp_name}')

    # Go to correct folder
    # dir_name = f'../experiments/{exp_name}' 
    # if not os.path.exists(dir_name):
    #   os.mkdir(dir_name)
    #   gitclone(dir_name)
    # gitupdate(dir_name)


    # # Task
    run_command(exp_id, dataset) # Change this
    # time.sleep(3)
    print(f'----task{exp_name} finishes')
    

def attach_exp_id(input_tuple, tuple_id):
    # Add experiment id in front of the input hyperparam tuple
    record = [tuple_id]
    record.append(input_tuple)
    return record




if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Change this
    # Hyperparms
    dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290'] 
    dataset_list = ['2258'] 

    input_list = []
    for id, dataset in enumerate(dataset_list):
        data_pair = attach_exp_id(dataset, id)
        input_list.append(data_pair)

    with Pool(processes = 9) as pool:
            pool.starmap(run, input_list)

    pool.join()
    print(f'all tasks finish')

