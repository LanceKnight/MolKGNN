import multiprocessing as mp
from multiprocessing import Pool, Value
import os
import os.path as osp
from tqdm import tqdm
import shutil, errno
import itertools
import time

branch = 'dimenet_pp' # Change this

def gitclone(dir_name):
    cwd = os.getcwd()
    os.chdir(dir_name)
    os.system('git clone git@github.com:LanceKnight/kgnn.git')
    os.chdir('kgnn')
    os.system(f'git checkout {branch}') 
    os.chdir(cwd)

def gitupdate():
    os.system('git gc')
    os.system(f'git checkout {branch}') 
    os.system('git pull')

def run_command(dataset): # Change this
    cwd = os.getcwd()
    print(f'dataset:{dataset}')
    # Model=kgnn
    if not osp.exists('logs/best_test_sample_scores.log'):
        os.system(f'python -W ignore entry.py \
            --task_name test_dimenetpp\
            --dataset_name {dataset} \
            --seed 42\
            --num_workers 16 \
            --dataset_path ../../../dataset/ \
            --enable_oversampling_with_replacement \
            --warmup_iterations 300 \
            --max_epochs 20\
            --peak_lr 1e-4 \
            --end_lr 1e-9 \
            --batch_size 17 \
            --default_root_dir actual_training_checkpoints \
            --gpus 1 \
            --test\
            ')


def run(folder):
    cwd = os.getcwd()
    os.chdir(osp.join(folder, 'kgnn'))
    gitupdate()

    folder_name_components = folder.split('_')
    dataset = folder_name_components[2][7:]
    run_command(dataset)



if __name__ == '__main__':
    start_time = time.time()
    mp.set_start_method('spawn')
    exp_dir = '/home/liuy69/projects/unified_framework/experiments/'

   # Get a list of folders
    folder_list = []
    for folder in os.listdir(exp_dir):
        if 'exp' in folder:
            folder_list.append(osp.join(exp_dir, folder))

    # Update git and run testing
    with Pool(processes = 9) as pool:
        pool.map(run, folder_list)

    # Gather testing results
    file_name = 'logs/all_test_result.log'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as output_file:
        for folder in folder_list:
            print('\n=======\n')
            with open(osp.join(exp_dir,f'{folder}/kgnn/logs/test_result.log'), 'r') as in_file:
                for line in in_file:
                    if 'Namespace' in line: # for arguments
                        components = line.split(', ')
                        for component in components:
                            if ('peak' in component) or ('layer') in component: # specifiy which arguments to print
                                out_content = component
                                print(out_content)
                                output_file.write(out_content)
                    else:
                        out_content = line
                        print(out_content)
                        output_file.write(out_content)
                output_file.write('\n=======\n')
    end_time=time.time()
    run_time = end_time-start_time
    print(f'finish getting all test result: {run_time/3600:0.0f}h{(run_time)%3600/60:0.0f}m{run_time%60:0.0f}')
