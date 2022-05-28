import multiprocessing as mp
from multiprocessing import Pool, Value
import os
from tqdm import tqdm
import shutil, errno
import itertools
import time
from datetime import datetime

branch = 'main' # Change this

def gitclone(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name)
    os.system('git clone git@github.com:LanceKnight/kgnn.git')
    os.chdir('kgnn')
    os.system(f'git checkout {branch}') 
    os.chdir(cwd)

def gitupdate(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name+'/kgnn')
    os.system('git gc')
    os.system(f'git checkout {branch}') 
    os.system('git pull')
    os.chdir(cwd)

def run_command(exp_id, args): 
    # Model=kgnn
    os.system(f'python -W ignore entry.py \
        --task_name experiments{exp_id}\
        --dataset_name {args[0]} \
        --seed {args[1]}\
        --num_workers 11 \
        --dataset_path ../../../dataset/ \
        --enable_oversampling_with_replacement \
        --warmup_iterations {args[2]} \
        --max_epochs {args[3]}\
        --peak_lr {args[4]} \
        --end_lr {args[5]} \
        --default_root_dir actual_training_checkpoints \
        --gpus 1 \
        --num_layers {args[6]} \
        --num_kernel1_1hop {args[7]} \
        --num_kernel2_1hop {args[8]} \
        --num_kernel3_1hop {args[9]} \
        --num_kernel4_1hop {args[10]} \
        --num_kernel1_Nhop {args[7]} \
        --num_kernel2_Nhop {args[8]} \
        --num_kernel3_Nhop {args[9]} \
        --num_kernel4_Nhop {args[10]} \
        --node_feature_dim 27 \
        --edge_feature_dim 7 \
        --hidden_dim {args[11]}\
        --batch_size {args[12]}\
        ')\

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

def run(exp_id, *args):
    exp_name = f'exp{exp_id}_full_param1' # Change this
    print(f'=====running {exp_name}')

    # Go to correct folder
    dir_name = f'../experiments/{exp_name}' 
    # if not os.path.exists(dir_name):
    #   os.mkdir(dir_name)
    #   gitclone(dir_name)
    # gitupdate(dir_name)

    global github_repo_dir
    copyanything(github_repo_dir, dir_name)
    cwd = os.getcwd()
    os.chdir(dir_name+'/kgnn')

    # # Task
    run_command(exp_id, args)
    # time.sleep(3)
    print(f'----{exp_name} finishes')
    os.chdir(cwd)
    

def attach_exp_id(input_tuple, tuple_id):
    # Add experiment id in front of the input hyperparam tuple
    record = [tuple_id]
    record.extend(list(input_tuple))
    return record


# Global variable
# Github repo template
github_repo_dir = f'../experiments/template_dataset_layers'# Change this

if __name__ == '__main__':
    mp.set_start_method('spawn')

    start_time = time.time()
    now = datetime.now()
    print(f'start time:{now}')

    # Change this
    # Hyperparms
    #dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
    dataset_list = [ '9999' ] # arg0
    seed_list = [42, 26, 30] # arg1
    warmup_list = [200] # arg2
    epochs_list = [20] # arg3
    peak_lr_list = [5e-1] # arg4
    end_lr_list = [1e-10] # arg5
    num_layer_list = [1] # arg6
    kernel1_list = [10] # arg7
    kernel2_list = [20] # arg8
    kernel3_list = [30] # arg9
    kernel4_list = [50] # arg10
    hidden_dim = [32] # arg11
    batch_size = [17] # arg12
    data_pair = list(itertools.product(dataset_list, seed_list, warmup_list, epochs_list, peak_lr_list, end_lr_list, num_layer_list, kernel1_list, kernel2_list, kernel3_list, kernel4_list, hidden_dim, batch_size )) 
    print(f'num data_pair:{len(data_pair)}')
    data_pair_with_exp_id = list(map(attach_exp_id, data_pair, range(len(data_pair))))
    print(f'data_pair_with_exp_id:{data_pair_with_exp_id}')

    file_name='utils/scheduler.log'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as out_file:
        out_file.write(f'num data_pair:{len(data_pair)}\n\n')
        out_file.write(f'data_pair_with_exp_id:{data_pair_with_exp_id}')


    # Clone once from github
    if not os.path.exists(github_repo_dir):
        os.mkdir(github_repo_dir)
        gitclone(github_repo_dir)
    gitupdate(github_repo_dir)

    
    with Pool(processes = 3) as pool:
        pool.starmap(run, data_pair_with_exp_id)

    end_time=time.time()
    run_time = end_time-start_time
    print(f'finish getting all test result: {run_time/3600:0.0f}h{(run_time)%3600/60:0.0f}m{run_time%60:0.0f}')
    now = datetime.now()
    print(f'finsh time:{now}')

