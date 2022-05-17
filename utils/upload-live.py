# import required module
import os
from clearml import Task
import multiprocessing as mp
from multiprocessing import Pool
import random

def run(file):
    tag = random.randint(100, 999)
    print(f'*uploading tag-{tag}-{os.path.basename(file)}...')
    Task.import_offline_session(session_folder_zip=file)
    print(f'~tag-{tag}-{os.path.basename(file)}')


if __name__ == '__main__':
    # Assign directories
    clearml_directory = '/home/live-lab/.clearml/cache/offline/'
    exp_directory = '/home/live-lab/projects/unified_framework/experiments/'

    # Set multiprocessing start method
    mp.set_start_method('spawn')

    # Iterate over task_info file in each experiments and form a list of file for uploading
    file_list = []
    for filename in os.listdir(exp_directory):
        if "exp" in filename:
            task_info_file = os.path.join(exp_directory, filename+'/kgnn/task_info')
            try:
                with open(task_info_file) as in_file:
                    task_id = in_file.readline().split(':')[1][:-1]
                    run_time = in_file.readline()
                    print(f'{filename} {run_time}')
                    zip_file_name = task_id+'.zip'
                    f = os.path.join(clearml_directory, zip_file_name)
                    # checking if it is a file
                    print(f)
                    if os.path.isfile(f):
                        file_list.append(f)
            except:
                print(f'{filename} does not have task_info')
                
    print(f'==== list of files to upload ====')
    for file in file_list:
        print(file)

    with Pool(processes = 9) as pool:
        pool.map(run, file_list)






