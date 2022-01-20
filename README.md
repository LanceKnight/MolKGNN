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

`python entry.py ---dataset_name CHIRAL1 --input_dim 5 --hidden_dim 32 --output_dim 1 --warmup_iterations 600 --tot_iterations 10000 --peak_lr 2e-4 --end_lr 1e-9 --batch_size 64 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3`
