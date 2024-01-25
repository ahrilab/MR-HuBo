#!/bin/bash

# Get the robot name and check if it is valid
robot_name=$1
valid_robot_names=("COMAN")

if [[ ! " ${valid_robot_names[@]} " =~ " ${robot_name} " ]]; then
    echo "Invalid robot name. Please choose between 'COMAN'."
    exit 1
fi

echo "Rendering motions with Pybullet for $robot_name..."

# directory & file format information
output_ext=$2
input_dir="out/pred_motions/$robot_name/new"
output_dir="out/pybullet/$robot_name"

# create an output directory if it does not exist
mkdir -p $output_dir

# Run the rendering script for each motion file
for file in "$input_dir"/*
do
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"

    echo "Rendering motions of $robot_name for $filename."
    python src/visualize/pybullet_render.py -v front --fps 120 -s -rp $input_dir/$filename.pkl -op $output_dir/$filename.$output_ext
done
