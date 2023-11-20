#!/bin/bash

# Define the source and target directories
RMD_DIR="./rmarkdown_files"
NOTEBOOKS_DIR="./notebooks"

# Create the target directory if it doesn't exist
mkdir -p "$NOTEBOOKS_DIR"

echo "Converting R Markdown files in $RMD_DIR..."

# Loop through each R Markdown file in the rmarkdown_files directory
for rmd_file in "$RMD_DIR"/*.Rmd; do
    # Extract the base filename without extension
    base_name=$(basename "$rmd_file" .Rmd)

    # Convert the R Markdown file to a Jupyter notebook
    jupytext --to ipynb --output "$NOTEBOOKS_DIR/${base_name}.ipynb" "$rmd_file"
done

echo "Conversion complete. Jupyter notebooks saved in $NOTEBOOKS_DIR."
