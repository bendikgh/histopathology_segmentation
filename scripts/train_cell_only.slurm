#!/bin/sh
#SBATCH --job-name=ocelot_cell_only_training     # Name for the job
#SBATCH --account=ie-idi                # Billing account
#SBATCH --time=0-03:00:00               # 0 days and 15 minutes limit

#SBATCH --partition=GPUQ                # Whether you need GPUs or CPUs
#SBATCH --gres=gpu:a100                    # Number of GPUs 
#SBATCH --nodes=1                       # 1 compute nodes
#SBATCH --mem=32G                        # 8GB memory

#SBATCH --output=output.txt             # Log file
#SBATCH --error=output.err              # Error file

WORKDIR=/cluster/work/jssaethe/histopathology_segmentation
cd ${WORKDIR}

echo "Job was submitted from this directory: $SLURM_SUBMIT_DIR."
echo "The name of the job is: $SLURM_JOB_NAME."
echo "The job ID is $SLURM_JOB_ID."
echo "The job was run on these nodes: $SLURM_JOB_NODELIST."

module purge 
module load Anaconda3/2022.10
conda activate specialization_project

python src/training_cell_only.py --data-dir /cluster/projects/vc/data/mic/open/OCELOT/ocelot_data --batch-size 8 --checkpoint-interval 10 --epochs 100
