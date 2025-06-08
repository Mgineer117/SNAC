#!/bin/bash
#SBATCH --job-name=snac
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --time=3-00:00:00
#SBATCH --output=snac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu


# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method crs --gpu-idx 1 --num-options 4 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method cvs --gpu-idx 1 --num-options 4 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method crvs --gpu-idx 1 --num-options 4 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method trs --gpu-idx 1 --num-options 4 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=1 nohup python3 main.py --env-name CtF --project CtF-SNAC --algo-name SNAC --method trs --gpu-idx 0 --num-options 4 --ctf-type strategic --episode-len 300 & & 
sleep 300


CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method crs --gpu-idx 1 --num-options 8 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method cvs --gpu-idx 1 --num-options 8 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method crvs --gpu-idx 1 --num-options 8 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name CtF --project CtF-SNAC --algo-name EigenOption --method trs --gpu-idx 1 --num-options 8 --ctf-type strategic --episode-len 300 & 
sleep 300
CUDA_VISIBLE_DEVICES=2 nohup python3 main.py --env-name CtF --project CtF-SNAC --algo-name SNAC --method trs --gpu-idx 0 --num-options 8 --ctf-type strategic --episode-len 300 & & 

# === Wait for all background jobs to finish ===
wait