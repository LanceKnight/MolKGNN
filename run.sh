[ -z "${exp_name}" ] && exp_name="qsar"
[ -z "${seed}" ] && seed="1"
[ -z "${arch}" ] && arch="--ffn_dim 768 --hidden_dim 768 --dropout_rate 0.1 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 5"
[ -z "${batch_size}" ] && batch_size="2" #"256"
[ -z "${dataset}" ] && dataset="435034"
# [ -z "${dataset}" ] && dataset="PCQM4M-LSC"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "exp_name: ${exp_name}"
echo "arch: ${arch}"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "==============================================================================="

default_root_dir="root_dir_5updates"
mkdir -p $default_root_dir
n_gpu=$(nvidia-smi -L | wc -l)

 
python entry.py --num_workers 8 --seed $seed --batch_size $batch_size \
      --dataset_name $dataset \
      --gpus $n_gpu --accelerator ddp --precision 16 --gradient_clip_val 5.0 \
      $arch \
      --default_root_dir $default_root_dir --warmup_updates 200 --tot_updates 500 --n_layers 2 --max_steps 70
