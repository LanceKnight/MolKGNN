# KGNN
# python entry.py --dataset_name 435034 --num_workers 2 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 10 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 17 --input_dim 32 --hidden_dim 32 --output_dim 1 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3 --num_kernel1_1hop 10 --num_kernel2_1hop 20 --num_kernel3_1hop 30 --num_kernel4_1hop 50 --num_kernel1_Nhop 10 --num_kernel2_Nhop 20 --num_kernel3_Nhop 30 --num_kernel4_Nhop 50 --machine ndslab2

# GCN
# python entry.py --dataset_name 435034 --num_workers 2 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 20 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 17 --input_dim 32 --hidden_dim 32 --output_dim 1 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3 --machine ndslab2

# ChebNet
# python entry.py --dataset_name 435034 --num_workers 2 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 20 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 17 --input_dim 32 --hidden_dim 32 --output_dim 1 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3 --K 10 --machine ndslab2

# DimeNet
# python entry.py --dataset_name 435034 --num_workers 2 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 20 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 17 --node_feature_dim 27 --hidden_dim 32 --output_dim 1 --default_root_dir actual_training_checkpoints --gpus 1 --num_layers 3

# ChIRoNet
python entry.py --dataset_name 435034 --num_workers 16 --dataset_path ../dataset/ --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 20 --peak_lr 5e-2 --end_lr 1e-9 --default_root_dir actual_training_checkpoints --gpus 1