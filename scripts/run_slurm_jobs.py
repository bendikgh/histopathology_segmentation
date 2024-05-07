import datetime
import os
import subprocess

from typing import List


def generate_python_arg_list(
    python_file: str,
    model_architecture: str,
    epochs,
    batch_size,
    checkpoint_interval,
    backbone,
    dropout,
    learning_rate,
    learning_rate_end,
    pretrained,
    warmup_epochs,
    do_save,
    do_eval,
    break_early,
    normalization,
    pretrained_dataset,
    leak_labels,
    loss_function,
    cell_image_input_size,
    tissue_image_input_size,
    id_,
) -> List:
    # Return a list of command components instead of a single string
    return [
        "python",
        python_file,
        "--model-architecture",
        model_architecture,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--checkpoint-interval",
        str(checkpoint_interval),
        "--backbone",
        backbone,
        "--dropout",
        str(dropout),
        "--learning-rate",
        str(learning_rate),
        "--learning-rate-end",
        str(learning_rate_end),
        "--pretrained",
        str(pretrained),
        "--warmup-epochs",
        str(warmup_epochs),
        "--do-save",
        str(do_save),
        "--do-eval",
        str(do_eval),
        "--break-early",
        str(break_early),
        "--normalization",
        normalization,
        "--cell-image-input-size",
        str(cell_image_input_size),
        "--tissue-image-input-size",
        str(tissue_image_input_size),
        "--pretrained-dataset",
        pretrained_dataset,
        "--leak-labels",
        str(leak_labels),
        "--loss-function",
        loss_function,
        "--id",
        str(id_),
    ]


def generate_slurm_script(
    job_name: str,
    python_file: str,
    duration_str: str,
    work_dir: str,
    model_architecture: str,
    epochs,
    batch_size,
    checkpoint_interval,
    backbone,
    dropout,
    learning_rate,
    learning_rate_end,
    pretrained,
    warmup_epochs,
    do_save,
    do_eval,
    break_early,
    normalization,
    cell_image_input_size,
    tissue_image_input_size,
    pretrained_dataset,
    leak_labels,
    loss_function,
    id_,
):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    output_str = f"{current_time}/{job_name}_normalize_{normalization.replace('+', '_').replace(' ', '')}_id-{id_}"
    python_arg_list = generate_python_arg_list(
        python_file=python_file,
        model_architecture=model_architecture,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        backbone=backbone,
        dropout=dropout,
        learning_rate=learning_rate,
        learning_rate_end=learning_rate_end,
        pretrained=pretrained,
        warmup_epochs=warmup_epochs,
        do_save=do_save,
        do_eval=do_eval,
        break_early=break_early,
        normalization=normalization,
        cell_image_input_size=cell_image_input_size,
        tissue_image_input_size=tissue_image_input_size,
        pretrained_dataset=pretrained_dataset,
        leak_labels=leak_labels,
        loss_function=loss_function,
        id_=id_,
    )
    python_call = " ".join(python_arg_list)
    return f"""#!/bin/sh

#SBATCH --job-name={job_name}
#SBATCH --account=ie-idi               
#SBATCH --time={duration_str}               

#SBATCH --partition=GPUQ               
#SBATCH --gres=gpu:1              
#SBATCH --constraint=gpu32g
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

{python_call}"""


def main():
    # For adding gpu constraint:
    """
    #SBATCH --constraint="gpu32g|gpu40g|gpu80g"
    """

    run_slurm = False

    # General parameters
    job_name = "exp8-additive-2"
    python_file = "src/run_trainable.py"
    duration_str: str = "0-08:00:00"
    work_dir = os.getcwd()

    # Script-specific parameters
    # "segformer_cell_only", "segformer_tissue_branch", "segformer_cell_branch", "segformer_joint_pred2input", "segformer_sum_sharing", "deeplab_cell_only", "deeplab_tissue_cell", "vit_unet"
    model_architecture = "segformer_additive_joint_pred2decoder"
    epochs = 2
    batch_size = 2
    checkpoint_interval = 10
    backbone = "b3"
    dropout = 0.3
    learning_rate = 1e-4
    learning_rate_end = 2e-5
    pretrained = 1
    warmup_epochs = 0
    do_save = 0
    do_eval = 1
    break_early = 1
    id_ = 1
    normalization = "macenko"
    leak_labels = 0
    loss_function = "dice-ce"
    cell_image_input_size = 512
    tissue_image_input_size = 1024
    # Options: "dice", "dice-wrapper", "dice-ce", "dice-ce-wrapper"
    # Use "-wrapper" when training tissue-branch

    # SegFormer
    pretrained_dataset = "ade"

    for id_ in range(1, 2):
        script_contents = generate_slurm_script(
            job_name=job_name,
            python_file=python_file,
            duration_str=duration_str,
            work_dir=work_dir,
            model_architecture=model_architecture,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            backbone=backbone,
            dropout=dropout,
            learning_rate=learning_rate,
            learning_rate_end=learning_rate_end,
            pretrained=pretrained,
            warmup_epochs=warmup_epochs,
            do_save=do_save,
            do_eval=do_eval,
            break_early=break_early,
            normalization=normalization,
            pretrained_dataset=pretrained_dataset,
            cell_image_input_size=cell_image_input_size,
            tissue_image_input_size=tissue_image_input_size,
            leak_labels=leak_labels,
            loss_function=loss_function,
            id_=id_,
        )
        script_filename = f"scripts/slurm_script_id{id_}.sh"

        if run_slurm:
            with open(script_filename, "w") as script_file:
                script_file.write(script_contents)

            subprocess.run(["sbatch", script_filename])
            print(f"Submitted: {script_filename}")

            os.remove(script_filename)
            print(f"Deleted: {script_filename}")
        else:
            python_arg_list = generate_python_arg_list(
                python_file=python_file,
                model_architecture=model_architecture,
                epochs=epochs,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
                backbone=backbone,
                dropout=dropout,
                learning_rate=learning_rate,
                learning_rate_end=learning_rate_end,
                pretrained=pretrained,
                warmup_epochs=warmup_epochs,
                do_save=do_save,
                do_eval=do_eval,
                break_early=break_early,
                normalization=normalization,
                cell_image_input_size=cell_image_input_size,
                tissue_image_input_size=tissue_image_input_size,
                pretrained_dataset=pretrained_dataset,
                leak_labels=leak_labels,
                loss_function=loss_function,
                id_=id_,
            )
            print(f"Running:\n {' '.join(python_arg_list)}")
            subprocess.run(python_arg_list, capture_output=False)


if __name__ == "__main__":
    main()
