#!/bin/bash
#SBATCH --job-name=finetune_openvla_allex
#SBATCH --output=logs/finetune_openvla_allex_%j.out
#SBATCH --error=logs/finetune_openvla_allex_%j.err
#SBATCH --partition=your_partition_name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00  # 3 days, adjust as needed

# Load conda (adjust based on your cluster setup)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# Go to your project directory (adjust if needed)
cd /virtual_lab/rlwrld/jaehyun/openvla

# Run the command
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path /virtual_lab/rlwrld/jaehyun/datasets/openvla-7b \
    --data_root_dir /virtual_lab/rlwrld/jaehyun/datasets/allex_rlds \
    --dataset_name allex_cube \
    --run_root_dir openvla-allex-full \
    --adapter_tmp_dir adapter-tmp \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 3.5e-4 \
    --wandb_project openvla_allex \
    --wandb_entity jaehyunkang \
    --save_steps 1000 \
    --max_steps 10000