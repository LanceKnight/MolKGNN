import multiprocessing as mp
from multiprocessing import Pool, Value
import os
import os.path as osp
from tqdm import tqdm
import shutil, errno
import itertools
import time
from datetime import datetime
import math

branch = 'shrink-full-comp' # Change this
task_comment = '\" optimize4; new_data_split; full\"' # Change this


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
    os.system('git pull')
    os.system(f'git checkout {branch}')
    os.system('git pull')
    os.chdir(cwd)

def run_command(exp_id, args): 
    # Model=kgnn
    os.system(f'python -W ignore entry.py \
        --task_name id{exp_id}_lr{args[4]}_L{args[6]}_seed{args[1]}\
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
        --node_feature_dim 28 \
        --edge_feature_dim 7 \
        --hidden_dim {args[11]}\
        --batch_size {args[12]}\
        --task_comment {task_comment}\
        --weight_decay {args[14]}\
        --dropout_ratio {args[15]}\
        ')\

def copyanything(src, dst):
    '''
    does not overwrite
    return True if created a new one
    return False if folder exist
    '''
    if os.path.exists(dst):
        # shutil.rmtree(dst)
        print(f'{dst} exists and remain untouched')
        return False
    else:
        try:
            shutil.copytree(src, dst)
        except OSError as exc: # python >2.5
            if exc.errno in (errno.ENOTDIR, errno.EINVAL):
                shutil.copy(src, dst)
            else: raise
        return True

def overwrite_dir(src, dst):
    '''
    copy and overwrite
    '''
    # If dst exits, remove it first
    if os.path.exists(dst):
        shutil.rmtree(dst)
        print(f'{dst} exists and overwritten')
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

def run(exp_id, *args):
    exp_name = f'exp{exp_id}_{args[0]}_seed{args[1]}_warm{args[2]}_epoch{args[3]}_peak{args[4]}_layers{args[6]}_k1{args[7]}_k2{args[8]}_k3{args[9]}_k4{args[10]}_hidden{args[11]}_batch{args[12]}_decay{args[14]}_dropout{args[15]}' # Change this
    print(f'=====running {exp_name}')

    # Go to correct folder
    dir_name = f'../experiments/{exp_name}' 
    # if not os.path.exists(dir_name):
    #   os.mkdir(dir_name)
    #   gitclone(dir_name)
    # gitupdate(dir_name)

    global github_repo_dir
    newly_created = copyanything(github_repo_dir, dir_name)
    cwd = os.getcwd()
    os.chdir(dir_name+'/kgnn')

    # # Task
    if not osp.exists('logs/test_sample_scores.log'):
        if not newly_created:
            os.chdir(cwd)
            overwrite_dir(github_repo_dir, dir_name)
            os.chdir(dir_name+'/kgnn') 
        os.makedirs('logs', exist_ok=True)
        with open('logs/params.log', 'w+') as out:
            out.write(f'dataset:{args[0]}')
            out.write(f'seed:{args[1]}')
            out.write(f'warmup:{args[2]}')
            out.write(f'epochs:{args[3]}')
            out.write(f'peak:{args[4]}')
            out.write(f'end:{args[5]}')
            out.write(f'layers:{args[6]}')
            out.write(f'kernel1:{args[7]}')
            out.write(f'kernel2:{args[8]}')
            out.write(f'kernel3:{args[9]}')
            out.write(f'kernel4:{args[10]}')
            out.write(f'hidden_dim:{args[11]}')
            out.write(f'batch_size:{args[12]}')
            out.write(f'trials:{args[13]}')
            out.write(f'weight decay:{args[14]}')
            out.write(f'droput:{args[15]}')
            out.write(f'{task_comment}')


        run_command(exp_id, args)
        # time.sleep(3)
        print(f'----{exp_name} finishes')
    else:
        print(f'----{exp_name} was done previously')
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
    print(f'scheduler start time:{now}')

    # Change this
    # Hyperparms
    # dataset_list = ['435008', '1798', '435034', '1843', '2258', '463087', '488997','2689', '485290']
    # dataset_list = ['463087','488997','2689', '485290', '1798']
    dataset_list = [ '2258' ] # arg0
    seed_list = [1, 2, 10] # arg1
    warmup_list = [300] # arg2
    epochs_list = [13] # arg3
    peak_lr_list = [5e-3] # arg4
    end_lr_list = [1e-10] # arg5
    num_layer_list = [4] # arg6
    kernel1_list = [10] # arg7
    kernel2_list = [20] # arg8
    kernel3_list = [30] # arg9
    kernel4_list = [50] # arg10
    hidden_dim = [32] # arg11
    batch_size = [16] # arg12
    trials=[0] # args13
    decay_list = [0.001] # arg14
    dropout_list = [0.2] # arg15
    data_pair = list(itertools.product(dataset_list, seed_list, warmup_list, epochs_list, peak_lr_list, end_lr_list, num_layer_list, kernel1_list, kernel2_list, kernel3_list, kernel4_list, hidden_dim, batch_size, trials, decay_list, dropout_list )) 
    print(f'num data_pair:{len(data_pair)}')
    data_pair_with_exp_id = list(map(attach_exp_id, data_pair, range(len(data_pair))))
    print(f'data_pair_with_exp_id:{data_pair_with_exp_id}')

    file_name='logs/scheduler.log'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as out_file:
        out_file.write(f'num data_pair:{len(data_pair)}\n\n')
        out_file.write(f'data_pair_with_exp_id:{data_pair_with_exp_id}')


    # Clone once from github
    if not os.path.exists(github_repo_dir):
        os.mkdir(github_repo_dir)
        gitclone(github_repo_dir)
    gitupdate(github_repo_dir)

    
    with Pool(processes = 7) as pool:
        pool.starmap(run, data_pair_with_exp_id)

    end_time=time.time()
    run_time = end_time-start_time
    run_time_str = f'run_time:{math.floor(run_time/3600)}h{math.floor((run_time)%3600/60)}m' \
                   f'{math.floor(run_time%60)}s'
    print(run_time_str)
    now = datetime.now()
    print(f'scheduler finsh time:{now}')

