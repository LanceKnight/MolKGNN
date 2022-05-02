import multiprocessing as mp
from multiprocessing import Pool
import os
from tqdm import tqdm
import shutil, errno
import itertools


def gitclone(dir_name):
	cwd = os.getcwd()
	os.chdir(dir_name)
	os.system('git clone git@github.com:LanceKnight/kgnn.git')
	os.chdir('kgnn')
	os.system('git checkout add-testing-set') # Change this
	os.chdir(cwd)

def gitupdate(dir_name):
	cwd = os.getcwd()
	os.chdir(dir_name+'/kgnn')
	os.system('git checkout add-testing-set')
	os.system('git pull')
	os.chdir(cwd)

def run_command(id, warmup, num_epochs, peak_lr, end_lr, num_layers):
	# Model=kgnn
	os.system(f'python entry.py \
		--task_name {id}\
		--dataset_name 435034 \
		--num_workers 16 \
		--dataset_path ../../../dataset/ \
		--enable_oversampling_with_replacement \
		--warmup_iterations {warmup} \
		--max_epochs {num_epochs} \
		--peak_lr {peak_lr} \
		--end_lr {end_lr} \
		--batch_size 17 \
		--default_root_dir actual_training_checkpoints \
		--gpus 1 \
		--num_layers {num_layers} \
		--num_kernel1_1hop 10 \
		--num_kernel2_1hop 20 \
		--num_kernel3_1hop 30 \
		--num_kernel4_1hop 50 \
		--num_kernel1_Nhop 10 \
		--num_kernel2_Nhop 20 \
		--num_kernel3_Nhop 30 \
		--num_kernel4_Nhop 50 \
		--node_feature_dim 27 \
		--edge_feature_dim 7 \
		--hidden_dim 64')\

def run(warmup, num_epochs, peak_lr, end_lr, num_layers):
	print(f'running exp-{id}')
	id+=1

	# Go to correct folder
	dir_name = f'../experiments/exp_warmup_epochs_lr_layer_{id}' # Change this
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
		gitclone(dir_name)
	gitupdate(dir_name)
	os.chdir(dir_name+'/kgnn')

	# Task
	run_command(id, warmup, num_epochs, peak_lr, end_lr, num_layers)



if __name__ == '__main__':
	mp.set_start_method('spawn')
	id=0

	warmup = [200, 2000, 20000]
	num_epochs = [10, 20, 50]
	peak_lr = [5e-1, 5e-2, 5e-3]
	end_lr = [1e-7, 1e-8, 1e-9, 1e-10]
	num_layers = [3, 4, 5]

	data_pair = list(itertools.product(warmup, num_epochs, peak_lr, end_lr, num_layers))
	print(f'num data_pair:{len(data_pair)}')
	print(f'data_pair:{data_pair}')

	
	with Pool(processes = 12) as pool:
		pool.starmap(run, data_pair)

	pool.join()
	print(f'finish')

