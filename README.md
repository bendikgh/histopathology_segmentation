# Histopathology Segmentation
Authors: Jarl Sondre Bringslid Sæther and Bendik Gjermundrød Holter

This repository contains the code used for our Master's thesis at the Norwegian 
University of Science and Technology (NTNU), with the title
"Multi-Level Magnification with Transformer-Based Architectures for Enhanced Cell 
Detection and Classification in Computational Pathology". The work is also presented
in the NIK track of NIKT 2024 with the title, "Enhancing Cell Detection with 
Transformer-Based Architectures in Multi-Level Magnification Classification for 
Computational Pathology".

## Running the experiments
Our experiments were run using NTNU's HPC cluster, IDUN, using the SLURM workload 
manager. To automate the creation of SLURM scripts we used the 
[run_slurm_jobs.py](scripts/run_slurm_jobs.py) file. 

# Useful Commands

## Converting notebooks
Since Git doesn't work well with notebooks, we include a script for converting .rmd 
files into .ipynb files and vice versa. This is done with the 
`create_jupyter_notebooks.sh` and `create_rmarkdown.sh` files, respectively. Before 
doing this, however, make sure you have permission to execute the scripts. This can be 
done by typing 
```
chmod +x scripts/create_jupyter_notebooks.sh
chmod +x scripts/create_rmarkdown.sh
```
Once you have the permission to run the scripts, you can convert .rmd files into 
notebooks (when you pull from Git) by typing 
```
./scripts/create_jupyter_notebooks.sh
```
and convert .ipynb files into rmarkdown (before you push to Git e.g.) by typing 
```
./scripts/create_rmarkdown.sh
```

## Creating requirements.txt file when using conda
If you are using Conda to manage your packages, you have probably noticed that typing
```
pip freeze > requirements.txt
```
does not work very well. A workaround with conda is to type the following command: 
```
pip list | awk 'NR>2 {print $1"=="$2}' > requirements.txt
```
This will create a `requirements.txt` file in an acceptable format so that you can 
run `pip install` with it. 
