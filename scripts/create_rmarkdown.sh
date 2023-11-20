#!/bin/bash

# Define the source and target directories
NOTEBOOKS_DIR="./notebooks"
RMD_DIR="./rmarkdown_files"

# Create the target directory if it doesn't exist
mkdir -p "$RMD_DIR"

echo "Converting notebooks in $NOTEBOOKS_DIR..."

# Loop through each Jupyter notebook in the notebooks directory
for notebook in "$NOTEBOOKS_DIR"/*.ipynb; do
    # Extract the base filename without extension
    base_name=$(basename "$notebook" .ipynb)

    # Convert the notebook to R Markdown format
    jupytext --to Rmd --output "$RMD_DIR/${base_name}.Rmd" "$notebook"
done

echo "Conversion complete. R Markdown files saved in $RMD_DIR."
