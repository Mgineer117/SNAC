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
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method crs  --num-options 8 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method cvs  --num-options 8 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method top  --num-options 8 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method trs  --num-options 8 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method trs  --num-options 8 --ctf-type regular --rendering & 
sleep 300


CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method crs  --num-options 10 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method cvs  --num-options 10 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method top  --num-options 10 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name EigenOption --method trs  --num-options 10 --ctf-type regular --rendering & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF --algo-name SNAC --method trs  --num-options 10 --ctf-type regular --rendering & 

# === Wait for all background jobs to finish ===
wait