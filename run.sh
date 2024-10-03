#!/bin/bash
#SBATCH --account=huytran1-ae-eng    # Account to be charged
#SBATCH --job-name=SNAC                  # Job name
#SBATCH --nodes=1                        # Number of machines
#SBATCH --ntasks=4                       # Number of CPU cores
#SBATCH --gres=gpu:1                 # Request GPU (Type:number)
#SBATCH --mem-per-cpu=64000              # Memory per node
#SBATCH --partition=eng-research-gpu               # Time limit hrs:min:sec
#SBATCH --output=multi-serial.o%j        # Name of batch job output file
##SBATCH --error=python_job.e%j          # Name of batch job error file
##SBATCH --mail-user=minjae5@illinois.edu  # Send email notifications
##SBATCH --mail-type=BEGIN,END           # Type of email notifications to send


# Load necessary modules (adjust based on your cluster environment)
module load cuda
module load cudnn

# Run your application (adjust as needed)
conda activate snac
python main.py "$@"