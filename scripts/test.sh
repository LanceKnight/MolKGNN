#!/bin/bash
set -e 

# for seed in 1798 1843 2258 2689 9999 435008 435034 463087 485290 488997
# do
seed=1798
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python entry.py --test --dataset_name ${seed} --gnn_type "kgnn" --dataset_path dataset/ \
--num_workers 0 --accelerator gpu --devices 1 \
--enable_oversampling_with_replacement --warmup_iterations 300 --max_epochs 40 --peak_lr 5e-3 \
--end_lr 1e-10 --batch_size 16 --default_root_dir training_molkgnn --num_layers 3 \
--num_kernel1_1hop 10 --num_kernel2_1hop 20 --num_kernel3_1hop 30 --num_kernel4_1hop 50 \
--num_kernel1_Nhop 10 --num_kernel2_Nhop 20 --num_kernel3_Nhop 30 --num_kernel4_Nhop 50 \
--node_feature_dim 28 --edge_feature_dim 7 --hidden_dim 32 --seed 1 --task_comment "this is a train on ${seed}"
# done