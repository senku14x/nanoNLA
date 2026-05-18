#!/bin/bash
#SBATCH --job-name=qwen3_ar_stage2
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/nla-experiments/logs/%x_%j.out

bash /workspace-vast/celeste/nla-experiments/launch/run_stage2_ar_parallel.sh
