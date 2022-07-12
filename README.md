# kgnn
A kernel-base 3D GNN for molecular representation learning

<!-- # dataset
This repo does NOT include the dataset. Please download the dataset from https://github.com/PattanaikL/chiral_gnn/tree/cda134523996d26f94f4c92ffad8c373d79731a0/data/d4_docking -->

# how to process datasets

`python dataset_multigenerator.py`

# how to run
uncompress the 9999.tar.gz file and you will get a folder named kgnn-based-9999-3D. Put the folder and the other two files in this structure.

<pre>
root_dir

|--dataset

|  |--qsar
|  |  |--clean_sdf
|  |  |  |--processed
|  |  |  |  |--kgnn-based-9999-3D
|  |  |  |--raw
|  |  |  |  |--9999_actives_new.sdf
|  |  |  |  |--9999_inactives_new.sdf
  
|--kgnn

  |--entry.py

  |--*.py

</pre>

Here is an exmaple for running the code:

`python entry.py --dataset_name 9999 --num_workers 16 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 3 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 16 --default_root_dir actual_training_checkpoints --num_layers 3 --num_kernel1_1hop 10 --num_kernel2_1hop 20 --num_kernel3_1hop 30 --num_kernel4_1hop 50 --num_kernel1_Nhop 10 --num_kernel2_Nhop 20 --num_kernel3_Nhop 30 --num_kernel4_Nhop 50 --node_feature_dim 27 --edge_feature_dim 7 --hidden_dim 32 --seed 1 --task_comment "this is a test"`
