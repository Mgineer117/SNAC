#!/bin/bash
#SBATCH --job-name=snac
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --output=snac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu


# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method top --num-options 6 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method cvs --num-options 6 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method crs --num-options 6 &  


CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method top --num-options 8 & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method cvs --num-options 8 &  
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method crs --num-options 8 &  


CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method top --num-options 10 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method cvs --num-options 10 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name Ant --project SNAC --algo-name EigenOption --method crs --num-options 10 &   




# === Wait for all background jobs to finish ===
wait
