#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --account=huytran1-ic
#SBATCH --job-name="snac"
#SBATCH --partition=eng-research-gpu
#SBATCH --output=snac.o%j

# === Load Environment ===
module purge
module load anaconda3/2024.10
module load cuda/12.8
source $(conda info --base)/etc/profile.d/conda.sh
conda activate snac  # replace with your actual environment name

# === Run Jobs ===
for num_options in 4 6 8 10; do
    for method in top cvs trs crs; do
        python3 main.py --env-name FourRooms \
                        --project SNAC \
                        --algo-name EigenOption \
                        --method $method \
                        --gpu-idx 0 \
                        --num-options $num_options
    done
done
