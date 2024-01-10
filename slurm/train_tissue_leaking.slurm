#!/bin/sh

#SBATCH --job-name=ocelot_tissue_leaking_training    # Name for the job
#SBATCH --account=ie-idi                        # Billing account
#SBATCH --time=0-12:00:00                       # 0 days and 15 minutes limit

#SBATCH --partition=GPUQ                        # Whether you need GPUs or CPUs
#SBATCH --gres=gpu:a100                         # Number of GPUs 
#SBATCH --nodes=1                               # 1 compute nodes
#SBATCH --mem=32G                        

#SBATCH --output=outputs/logs/output_tissue_leaking-resnet34_dr-0.3_lr-1e-4.txt
#SBATCH --error=outputs/logs/output_tissue_leaking-resnet34_dr-0.3_lr-1e-4.err 

WORKDIR=/cluster/work/jssaethe/histopathology_segmentation
cd ${WORKDIR}

echo "Job was submitted from this directory: $SLURM_SUBMIT_DIR."
echo "The name of the job is: $SLURM_JOB_NAME."
echo "The job ID is $SLURM_JOB_ID."
echo "The job was run on these nodes: $SLURM_JOB_NODELIST."

module purge 
module load Anaconda3/2022.10
conda activate specialization_project

python src/training_tissue_leaking.py --data-dir /cluster/projects/vc/data/mic/open/OCELOT/ocelot_data --epochs 300 --batch-size 8 --checkpoint-interval 5 --backbone resnet34 --dropout 0.3 --learning-rate 1e-4