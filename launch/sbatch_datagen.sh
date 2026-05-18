#!/bin/bash
#SBATCH --job-name=qwen3_datagen
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

mkdir -p /workspace-vast/celeste/nla-experiments/logs
bash /workspace-vast/celeste/nla-experiments/launch/run_datagen.sh
