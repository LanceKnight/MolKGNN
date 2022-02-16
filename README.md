# kgnn
A kernel-base 3D GNN for molecular representation learning

# dataset
This repo does NOT include the dataset. Please download the dataset from https://github.com/PattanaikL/chiral_gnn/tree/cda134523996d26f94f4c92ffad8c373d79731a0/data/d4_docking

# how to run
put the dataset "d4_docking" folder one direcory above. i.e. the directory structure should be like this:

<pre>
root_dir

|--dataset

|  |--d4_docking

|--kgnn

  |--entry.py

  |--*.py

</pre>

Here is an exmaple for running the code:

`--dataset_name CHIRAL1 --input_dim 32 --hidden_dim 32 --output_dim 1 --warmup_iterations 200 --tot_iterations 1641 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 128 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 2 --num_workers 16 --num_kernel1_1hop 50 --num_kernel2_1hop 50 --num_kernel3_1hop 50 --num_kernel4_1hop 50 --num_kernel1_Nhop 50 --num_kernel2_Nhop 50 --num_kernel3_Nhop 50 --num_kernel4_Nhop 50`

if you have more than 1 gpus, you can set the gpus flag above to a different number
