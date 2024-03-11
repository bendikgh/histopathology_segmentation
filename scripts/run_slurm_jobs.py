import datetime
import os
import subprocess


def generate_slurm_script(
    epoch,
    batch_size,
    checkpoint_interval,
    backbone,
    dropout,
    learning_rate,
    pretrained,
    warmup_epochs,
    do_save,
    break_early,
    normalization,
    id_,
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""#!/bin/sh

#SBATCH --job-name=ocelot_cell_only_training_{normalization.replace("+","_").replace(" ", "")}_{id_}    
#SBATCH --account=ie-idi               
#SBATCH --time=0-06:00:00               

#SBATCH --partition=GPUQ               
#SBATCH --gres=gpu:a100                
#SBATCH --nodes=1                      
#SBATCH --mem=32G                        

#SBATCH --output=outputs/logs/output_deeplab_cell_only_normalize_{normalization.replace("+","_").replace(" ", "")}_{current_time}_id-{id_}.txt           
#SBATCH --error=outputs/logs/output_deeplab_cell_only_normalize_{normalization.replace("+","_").replace(" ", "")}_{current_time}_id-{id_}.err            

WORKDIR=/cluster/work/jssaethe/histopathology_segmentation
cd ${{WORKDIR}}

echo "Job was submitted from this directory: $SLURM_SUBMIT_DIR."
echo "The name of the job is: $SLURM_JOB_NAME."
echo "The job ID is $SLURM_JOB_ID."
echo "The job was run on these nodes: $SLURM_JOB_NODELIST."

module purge 
module load Anaconda3/2022.10
conda activate specialization_project

python src/training_cell_only.py \\
  --epochs {epoch} \\
  --batch-size {batch_size} \\
  --checkpoint-interval {checkpoint_interval} \\
  --backbone {backbone} \\
  --dropout {dropout} \\
  --learning-rate {learning_rate} \\
  --pretrained {pretrained} \\
  --warmup-epochs {warmup_epochs} \\
  --do-save {do_save} \\
  --break-early {break_early} \\
  --normalization "{normalization}" \\
  --id {id_}"""


def main():
    epochs = 100
    batch_size = 4
    checkpoint_interval = 30
    backbone = "resnet50"
    dropout = 0.3
    learning_rate = 1e-4
    pretrained = 1
    warmup_epochs = 10
    do_save = 1
    break_early = 0
    ids = ["1", "2", "3", "4", "5"]
    normalizations = [
        "off",
        "imagenet",
        "cell",
        "macenko",
        "macenko + cell",
        "macenko + imagenet",
    ]

    for normalization in normalizations:
        for id_ in ids:
            script_contents = generate_slurm_script(
                epochs,
                batch_size,
                checkpoint_interval,
                backbone,
                dropout,
                learning_rate,
                pretrained,
                warmup_epochs,
                do_save,
                break_early,
                normalization,
                id_,
            )
            script_filename = f"scripts/slurm_script_id{id_}.sh"
            with open(script_filename, "w") as script_file:
                script_file.write(script_contents)

            # Submit the script
            subprocess.run(["sbatch", script_filename])
            print(f"Submitted: {script_filename}")

            # Delete the script file
            os.remove(script_filename)
            print(f"Deleted: {script_filename}")


if __name__ == "__main__":
    main()
