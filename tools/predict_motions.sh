#!/bin/bash

# Get the robot name and check if it is valid
robot_name=$1
valid_robot_names=("REACHY" "COMAN" "NAO")

if [[ ! " ${valid_robot_names[@]} " =~ " ${robot_name} " ]]; then
    echo "Invalid robot name. Please choose between 'REACHY' and 'COMAN'."
    exit 1
fi

echo "Predicting motions for $robot_name"

# directory information
input_dir="data/gt_motions/amass_data"
output_dir="out/pred_motions/$robot_name"

# create an output directory if it does not exist
mkdir -p $output_dir

# Run the prediction script for each motion file
for file in "$input_dir"/*
do
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"

    echo "Generate predicted motions of $robot_name for $filename."
    python src/model/test_rep_only.py -r $robot_name -hp $input_dir/$filename.npz -rp $output_dir/rep_only_$filename.pkl
done
