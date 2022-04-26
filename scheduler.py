import multiprocessing as mp
from multiprocessing import Pool
import os
from tqdm import tqdm
import shutil, errno


def gitclone(dir_name):
	cwd = os.getcwd()
	os.chdir(dir_name)
	os.system('git clone git@github.com:LanceKnight/kgnn.git')
	os.chdir('kgnn')
	os.system('git checkout scheduler')
	os.chdir(cwd)

def run(batch_size):
	print(f'running batch_size={batch_size}')
	dir_name = f'../experiments/exp_batch_size{batch_size}'
	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
		gitclone(dir_name)

	os.chdir(dir_name+'/kgnn')

	# Model=kgnn
	os.system(f'python entry.py \
		--task_name batchsize{batch_size}\
		--dataset_name 435034 \
		--num_workers 16 \
		--dataset_path ../../../dataset/ \
		--enable_oversampling_with_replacement \
		--warmup_iterations 200 \
		--max_epochs 20 \
		--peak_lr 5e-2 \
		--end_lr 1e-9 \
		--batch_size {batch_size} \
		--default_root_dir actual_training_checkpoints \
		--gpus 1 \
		--num_layers 3 \
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


if __name__ == '__main__':
	mp.set_start_method('spawn')

	batch_list = [10, 20, 50, 100]
	data_pair = []
	id = 0
	for batch_size in batch_list:
			data_pair.append([batch_size])
	print(f'data_pair:{data_pair}')

	
	with Pool(processes = 10) as pool:
		pool.starmap(run, data_pair)

	pool.join()
	print(f'finish')

