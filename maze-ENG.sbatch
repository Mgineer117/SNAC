#!/bin/bash
#SBATCH --job-name=snac
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --time=2-00:00:00
#SBATCH --output=snac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu


# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method top --num-options 6 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method cvs --num-options 6 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method crvs --num-options 6 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method trvs --num-options 6 &  

CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method top --num-options 8 & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method cvs --num-options 8 &  
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method crvs --num-options 8 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method trvs --num-options 8 &  

CUDA_VISIBLE_DEVICES=4 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method top --num-options 10 & 
CUDA_VISIBLE_DEVICES=4 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method cvs --num-options 10 & 
CUDA_VISIBLE_DEVICES=5 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method crvs --num-options 10 &
CUDA_VISIBLE_DEVICES=5 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method trvs --num-options 10 &   

CUDA_VISIBLE_DEVICES=6 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method top --num-options 12 & 
CUDA_VISIBLE_DEVICES=6 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method cvs --num-options 12 & 
CUDA_VISIBLE_DEVICES=7 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method crvs --num-options 12 &
CUDA_VISIBLE_DEVICES=7 python3 main.py --env-name Maze --project SNAC --algo-name EigenOption --method trvs --num-options 12 &  


# === Wait for all background jobs to finish ===
wait
