#!/bin/bash
#SBATCH --job-name=snac
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=snac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate snac

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method top --snac-split-ratio 0.25 --num-options 8 --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method top --snac-split-ratio 0.5 --num-options 8 --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method top --snac-split-ratio 0.75 --num-options 8 --rendering & 
sleep 300

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method cvs --snac-split-ratio 0.25 --num-options 8 --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method cvs --snac-split-ratio 0.5 --num-options 8 --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method cvs --snac-split-ratio 0.75 --num-options 8 --rendering & 
sleep 300


CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method crs --snac-split-ratio 0.25 --num-options 8 --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method crs --snac-split-ratio 0.5 --num-options 8 --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method crs --snac-split-ratio 0.75 --num-options 8 --rendering & 
sleep 300

# === Wait for all background jobs to finish ===
wait