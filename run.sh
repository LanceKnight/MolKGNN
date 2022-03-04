# on Yu computer
# python entry.py --dataset_name CHIRAL1 --input_dim 32 --hidden_dim 32 --output_dim 1 --warmup_iterations 200 --max_epoch 50 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 256 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3 --num_workers 5 --num_kernel1_1hop 10 --num_kernel2_1hop 20 --num_kernel3_1hop 50 --num_kernel4_1hop 100 --num_kernel1_Nhop 10 --num_kernel2_Nhop 20 --num_kernel3_Nhop 50 --num_kernel4_Nhop 100


# on Lance computer
# python entry.py --dataset_name CHIRAL1 --input_dim 32 --hidden_dim 32 --output_dim 1 --warmup_iterations 200 --max_epoch 50 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 128 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 2 --num_workers 16 --num_kernel1_1hop 50 --num_kernel2_1hop 50 --num_kernel3_1hop 50 --num_kernel4_1hop 50 --num_kernel1_Nhop 50 --num_kernel2_Nhop 50 --num_kernel3_Nhop 50 --num_kernel4_Nhop 50

# Test
# python entry.py --dataset_name CHIRAL1 --input_dim 32 --hidden_dim 32 --output_dim 1 --warmup_iterations 200 --max_epoch 10 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 10 --default_root_dir temp_training_checkpoints --gpus 1 --num_layers 2 --num_workers 5 --num_kernel1_1hop 5 --num_kernel2_1hop 5 --num_kernel3_1hop 5 --num_kernel4_1hop 5 --num_kernel1_Nhop 5 --num_kernel2_Nhop 5 --num_kernel3_Nhop 5 --num_kernel4_Nhop 10
# 50 Kernel Test
python entry.py --dataset_name D4DCHP --input_dim 32 --hidden_dim 32 --output_dim 1 --warmup_iterations 200 --tot_iterations 10810 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 17 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3 --num_workers 3 --num_kernel1_1hop 10 --num_kernel2_1hop 20 --num_kernel3_1hop 30 --num_kernel4_1hop 50 --num_kernel1_Nhop 10 --num_kernel2_Nhop 20 --num_kernel3_Nhop 30 --num_kernel4_Nhop 50
