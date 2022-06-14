import multiprocessing as mp
from multiprocessing import Pool, Value
import os
import os.path as osp
from tqdm import tqdm
import shutil, errno
import itertools
import time
import pandas as pd

branch = 'main' # Change this

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
    use_best = True
    monitored_metric = 'logAUC'
    start_time = time.time()
    mp.set_start_method('spawn')
    exp_dir = '/home/live-lab/projects/unified_framework/experiments/'

   # Get a list of folders
    folder_list = []
    for folder in os.listdir(exp_dir):
        if 'exp' in folder:
            folder_list.append(osp.join(exp_dir, folder))

    # Update git and run testing
    # with Pool(processes = 9) as pool:
    #     pool.map(run, folder_list)

    # Gather testing results
    file_name = 'logs/all_test_result.log'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    out_table = {}
    with open(file_name, 'w') as output_file:
        for folder in folder_list:
            metric_counter = 0 # if ==0, last metric, if ==1, best metric
            base_name = osp.basename(folder)
            name_components = base_name.split('_')
            print(name_components)
            seed = name_components[3]
            peak = name_components[4]
            # layers = name_components[7]
            
            print('\n=======\n')
            try:
                with open(osp.join(exp_dir,f'{folder}/kgnn/logs/test_result.log'), 'r') as in_file:
                    for line in in_file:
                        if 'Namespace' in line: # for arguments
                            components = line.split(', ')
                            for component in components:
                                if ('seed' in component) or ('peak') in component: # specifiy which arguments to print
                                    print(component)
                                    split_component = component.split('=')
                                    component_name = split_component[0]
                                    component_value = split_component[1]
                                    
                                    out_content = out_table
                                    # print(out_content)
                                    # output_file.write(out_content)
                        else: # for metrics
                            if 'logAUC' in line:
                                print(f'peak_{peak};seed_{seed}')
                                print(f'{line}')
                                split_line = line.split(',')
                                loss = float(split_line[0].split(': ')[1]) # Get loss
                                ppv = float(split_line[1].split(': ')[1]) # Get ppv
                                logAUC = float(split_line[2].split(': ')[1])# Get logAUC
                                f1 = float(split_line[3].split(': ')[1].split('}')[0]) # Get f1
                                if monitored_metric == 'logAUC':
                                    metric = logAUC
                                elif monitored_metric == 'ppv':
                                    metric = ppv
                                elif monitored_metric == 'f1':
                                    metric = f1

                                if metric_counter == 0:
                                    key = 'last'
                                    if use_best == False:
                                        out_table.setdefault(f'{peak}',[]).append({f'{key}_{monitored_metric}_{seed}':f'{metric}'})
                                        out_content = out_table
                                        # print(out_content)
                                else:
                                    key = 'best'
                                    if use_best == True:
                                        out_table.setdefault(f'{peak}',[]).append({f'{key}_{monitored_metric}_{seed}':f'{metric}'})
                                        out_content = out_table
                                    # print(out_content)
                                metric_counter+=1
                            
                            
                            
                            # output_file.write(out_content)
                    output_file.write('\n=======\n')
            except Exception as e:
                key = 'best' if use_best else 'last'
                print(f'error message:{e}')
                print(f'error folder:{folder}')
                out_table.setdefault(f'{peak}',[]).append({f'{key}_{monitored_metric}_{seed}':f'None'})


        # for folder in folder_list:
        #     print('\n=======\n')
        #     with open(osp.join(exp_dir,f'{folder}/kgnn/logs/test_result.log'), 'r') as in_file:
        #         for line in in_file:
        #             if 'Namespace' in line: # for arguments
        #                 components = line.split(', ')
        #                 for component in components:
        #                     if ('peak' in component) or ('layer') in component: # specifiy which arguments to print
        #                         out_content = component
        #                         print(out_content)
        #                         output_file.write(out_content)
        #             else:
        #                 out_content = line
        #                 print(out_content)
        #                 output_file.write(out_content)
        #         output_file.write('\n=======\n')

    
    # Prepare dataframe
    sorted_key_list = []
    for peak_layer_comb, peak_layer_list  in out_table.items():
        print('====')
        print(f'comb:{peak_layer_comb}:')
        row_index = peak_layer_comb
        ori_sorted_list = sorted(peak_layer_list, key=lambda x:list(x.items())[0][0])
        sorted_list = list(map(lambda x: list(x.items())[0][1], ori_sorted_list))
        print(f'sorted_list:{sorted_list}')
        sorted_key_list = list(map(lambda x: list(x.items())[0][0], ori_sorted_list))
        print(f'sorted_key:{sorted_key_list}')
        out_table[peak_layer_comb] = sorted_list
        for each in sorted_list:
            print(each)
    

    sorted_out_table = out_table

    output_df = pd.DataFrame.from_dict(sorted_out_table, orient='index')
    output_df.columns=sorted_key_list
    print(output_df)
    output_df.to_csv('logs/all_test_result_df.csv')    
    print('\n')

    end_time=time.time()
    run_time = end_time-start_time
    print(f'finish getting all test result: {run_time/3600:0.0f}h{(run_time)%3600/60:0.0f}m{run_time%60:0.0f}')
