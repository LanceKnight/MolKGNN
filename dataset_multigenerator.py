import multiprocessing as mp
from multiprocessing import Pool, Value
import os
from tqdm import tqdm
import shutil, errno
import itertools
import time



def gitclone(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name)
    os.system('git clone git@github.com:LanceKnight/kgnn.git')
    os.chdir('kgnn')
    os.system('git checkout bcl-benchmark') # Change this
    os.chdir(cwd)

def gitupdate(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name+'/kgnn')
    os.system('git gc')
    os.system('git checkout bcl-benchmark') # Change this
    os.system('git pull')
    os.chdir(cwd)

def run_command(exp_id, dataset): # Change this
    # Model=kgnn
    os.system(f'python wrapper.py \
        --task_name {exp_id}\
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
    record.extend(list(input_tuple))
    return record


# Global variable
# Github repo template
github_repo_dir = f'../experiments/exp_template_dataset_layers'# Change this

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Change this
    # Hyperparms
    dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290'] 

    if not os.path.exists(github_repo_dir):
        os.mkdir(github_repo_dir)
        gitclone(github_repo_dir)
    gitupdate(github_repo_dir)

    with Pool(processes = 5) as pool:
        for dataset in dataset_list:
            pool.starmap(run, dataset)

    pool.join()
    print(f'finish')

