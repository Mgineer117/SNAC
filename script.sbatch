#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --account=huytran1-ic
#SBATCH --job-name="snac"
#SBATCH --partition=eng-research-gpu
#SBATCH --output=snac.o%j

# Email settings (you can set it to send emails when the job starts, finishes, or fails)
#SBATCH --mail-type=FAIL    # Can be 'FAIL', 'BEGIN', 'END', or 'ALL'
#SBATCH --mail-user=minjae5@illinois.edu  # Replace with your email

# === Load Cluster Modules ===
if [[ $- == *i* ]]; then
    module purge
    module load anaconda3/2024.10
    module load git/2.48.1
    module load cuda/12.8
    module load gcc/13.3.0
fi

# === Initialize Conda ===
source $(conda info --base)/etc/profile.d/conda.sh

# === Run Experiments ===
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method top --num-options 4
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method cvs --num-options 4
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method trs --num-options 4
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method crs --num-options 4

# Additional runs with varying options
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method top --num-options 6
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method cvs --num-options 6
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method trs --num-options 6
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method crs --num-options 6

python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method top --num-options 8
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method cvs --num-options 8
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method trs --num-options 8
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method crs --num-options 8

python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method top --num-options 10
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method cvs --num-options 10
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method trs --num-options 10
python3 main.py --env-name FourRooms --project SNAC --algo-name EigenOption --method crs --num-options 10
