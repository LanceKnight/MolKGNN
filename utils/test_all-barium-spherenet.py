import multiprocessing as mp
from multiprocessing import Pool, Value
import os
import os.path as osp
from tqdm import tqdm
import shutil, errno
import itertools
import time
import pandas as pd

branch = 'spherenet' # Change this
task_comment = '\" test all spherenet\"'

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
            --dataset_name {dataset} \
            --num_workers 11 \
            --max_epochs 1\
            --dataset_path ../../../dataset/ \
            --enable_oversampling_with_replacement \
            --batch_size 32 \
            --default_root_dir actual_training_checkpoints \
            --gpus 1 \
            --task_comment {task_comment}\
            --test\
            ')


def run(folder):
    cwd = os.getcwd()
    os.chdir(osp.join(folder, 'kgnn'))
    gitupdate()

    folder_name_components = folder.split('_')
    dataset = folder_name_components[2][7:]
    run_command(dataset)


def get_table(use_best, best_based_on, monitored_metric, folder_list):


    # Gather testing results
    file_name = 'logs/all_test_result.log'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    out_table = {}
    with open(file_name, 'w') as output_file:
        for folder in folder_list:
            metric_counter = 0 # if ==0, last metric, if ==1, best metric
            base_name = osp.basename(folder)
            name_components = base_name.split('_')
            exp_id = name_components[0]
            seed = name_components[3]
            peak = name_components[4]
            
            # print('\n=======\n')
            try:
                with open(osp.join(exp_dir,f'{folder}/kgnn/logs/test_result.log'), 'r') as in_file:
                    for line in in_file:
                        if 'Namespace' in line: # for arguments
                            components = line.split(', ')
                            for component in components:
                                if ('seed' in component) or ('peak') in component: # specifiy which arguments to print
                                    split_component = component.split('=')
                                    component_name = split_component[0]
                                    component_value = split_component[1]
                                    
                                    out_content = out_table
                                    # print(out_content)
                                    # output_file.write(out_content)
                        else: # for metrics
                            if line == ('best_'+best_based_on+":\n") or (best_based_on == 'last' and line == 'last:\n'):
                                if (line == 'last:\n'):
                                    is_last = True
                                else:
                                    is_last = False
                                line = next(in_file)
                                # print(f'id_{exp_id}_{peak};{layers};{seed}')
                                # print(f'{line}')
                                split_line = line.split(',')
                                loss = float(split_line[0].split(': ')[1]) # Get loss
                                ppv = float(split_line[1].split(': ')[1]) # Get ppv
                                logAUC_0_001_0_1 = float(split_line[2].split(': ')[1])# Get logAUC_0.001_0.1
                                logAUC_0_001_1 = float(split_line[3].split(': ')[1])# Get logAUC_0.001_1
                                f1 = float(split_line[4].split(': ')[1].split('}')[0]) # Get f1
                                AUC = float(split_line[5].split(': ')[1].split('}')[0]) # Get f1

                                if monitored_metric == 'logAUC_0.001_0.1':
                                    metric = logAUC_0_001_0_1
                                elif monitored_metric == 'ppv':
                                    metric = ppv
                                elif monitored_metric == 'f1':
                                    metric = f1
                                elif monitored_metric == 'logAUC_0.001_1':
                                    metric = logAUC_0_001_1
                                elif monitored_metric == 'AUC':
                                    metric = AUC
                                elif monitored_metric == 'loss':
                                    metric = loss
                                

                                if is_last:
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
                                # metric_counter+=1
                            
                            
                            
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
        # print('====')
        # print(f'comb:{peak_layer_comb}:')
        row_index = peak_layer_comb
        ori_sorted_list = sorted(peak_layer_list, key=lambda x:list(x.items())[0][0])
        sorted_list = list(map(lambda x: list(x.items())[0][1], ori_sorted_list))
        # print(f'sorted_list:{sorted_list}')
        sorted_key_list = list(map(lambda x: list(x.items())[0][0], ori_sorted_list))
        # print(f'sorted_key:{sorted_key_list}')
        out_table[peak_layer_comb] = sorted_list
        # for each in sorted_list:
        #     print(each)
    

    sorted_out_table = out_table

    output_df = pd.DataFrame.from_dict(sorted_out_table, orient='index')
    output_df.columns=sorted_key_list
    return output_df


if __name__ == '__main__':
    use_best = True
    best_based_on = 'logAUC_0.001_0.1'
    # best_based_on = 'AUC'
    monitored_metrics = ['AUC', 'f1', 'logAUC_0.001_0.1', 'logAUC_0.001_1', 'loss', 'ppv']
    start_time = time.time()
    mp.set_start_method('spawn')

    exp_dir = os.getcwd()#'/home/liuy69/projects/unified_framework/experiments/'

   # Get a list of folders
    folder_list = []
    for folder in os.listdir(exp_dir):
        if 'exp' in folder:
            folder_list.append(osp.join(exp_dir, folder))

    # check if all has logs/test_results.log
    result_exists = True
    print(f'check folders{folder_list}')
    for folder in folder_list:
        print(f'checking {folder}')
        if not os.path.exists(f'{folder}/kgnn/logs/test_results.log'):
            print(f'{folder} does not have result')
            result_exists = False
            break

    # Update git and run testing
    if result_exists:
        print(f'all folders has results')
    else:
        print(f'at least one folder does not have test results')
        with Pool(processes = 3) as pool:
            pool.map(run, folder_list)

    all_table = pd.DataFrame()

    for monitored_metric in monitored_metrics:
        print(f'metric:{monitored_metric}')
        output_df = get_table(use_best, best_based_on, monitored_metric, folder_list)
        # print(output_df)
        all_table = pd.concat([all_table, output_df], axis = 1)
        output_df.to_csv(f'logs/all_test_result_df_{monitored_metric}.csv')    
        print('\n')

    print(all_table)
    all_table.to_csv(f'logs/all_test_result_df_all_table.csv')
    end_time=time.time()
    run_time = end_time-start_time
    print(f'finish getting all test result: {run_time/3600:0.0f}h{(run_time)%3600/60:0.0f}m{run_time%60:0.0f}')