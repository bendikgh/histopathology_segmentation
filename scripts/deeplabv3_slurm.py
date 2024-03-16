import datetime
import os
import subprocess


def generate_slurm_script(
    job_name: str,
    python_file: str,
    duration_str: str,
    work_dir: str,
    epochs,
    batch_size,
    checkpoint_interval,
    backbone,
    dropout,
    learning_rate,
    pretrained,
    warmup_epochs,
    do_save,
    do_eval,
    break_early,
    normalization,
    id_,
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    output_str = f"outputs/logs/{current_time}/{job_name}_normalize_{normalization.replace('+', '_').replace(' ', '')}_id-{id_}"
    return f"""#!/bin/sh

#SBATCH --job-name={job_name}
#SBATCH --account=ie-idi               
#SBATCH --time={duration_str}               

#SBATCH --partition=GPUQ               
#SBATCH --gres=gpu:a100                
#SBATCH --nodes=1                      
#SBATCH --mem=32G                        

#SBATCH --output=outputs/logs/{output_str}.txt           
#SBATCH --error=outputs/logs/{output_str}.err            

cd {work_dir}

echo "Job was submitted from this directory: $SLURM_SUBMIT_DIR."
echo "The name of the job is: $SLURM_JOB_NAME."
echo "The job ID is $SLURM_JOB_ID."
echo "The job was run on these nodes: $SLURM_JOB_NODELIST."

module purge 
module load Anaconda3/2022.10
conda activate specialization_project

python {python_file} \\
  --epochs {epochs} \\
  --batch-size {batch_size} \\
  --checkpoint-interval {checkpoint_interval} \\
  --backbone {backbone} \\
  --dropout {dropout} \\
  --learning-rate {learning_rate} \\
  --pretrained {pretrained} \\
  --warmup-epochs {warmup_epochs} \\
  --do-save {do_save} \\
  --do-eval {do_eval} \\
  --break-early {break_early} \\
  --normalization "{normalization}" \\
  --id {id_}"""


def main():
    run_script = False

    # General parameters
    job_name = "ocelot_deeplab_cell_only"
    python_file = "src/train_cell_branch.py"
    duration_str: str = "0-06:00:00"
    work_dir = os.getcwd()

    # Script-specific parameters
    epochs = 1
    batch_size = 2
    checkpoint_interval = 30
    backbone = "resnet50"
    dropout = 0.3
    learning_rate = 1e-4
    pretrained = 1
    warmup_epochs = 0
    do_save = 1
    do_eval = 1
    break_early = 1
    id_ = 1
    normalization = "macenko"

    script_contents = generate_slurm_script(
        job_name=job_name,
        python_file=python_file,
        duration_str=duration_str,
        work_dir=work_dir,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        backbone=backbone,
        dropout=dropout,
        learning_rate=learning_rate,
        pretrained=pretrained,
        warmup_epochs=warmup_epochs,
        do_save=do_save,
        do_eval=do_eval,
        break_early=break_early,
        normalization=normalization,
        id_=id_,
    )
    script_filename = f"scripts/slurm_script_id{id_}.sh"
    with open(script_filename, "w") as script_file:
        script_file.write(script_contents)

    if run_script:
        # Submit the script
        subprocess.run(["sbatch", script_filename])
        print(f"Submitted: {script_filename}")
    else:
        print(script_contents)

    # Delete the script file
    os.remove(script_filename)
    print(f"Deleted: {script_filename}")


if __name__ == "__main__":
    main()
