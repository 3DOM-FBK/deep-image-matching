#!/bin/bash

# Set the data directory
DATA_DIR=data
DATASET=belv_lingua_easy

INPUT_DIR=$DATA_DIR/$DATASET

# Run SfM with superpoint+superglue
python ./main.py --config superpoint+lightglue --images $INPUT_DIR --strategy bruteforce --force -V

# # Run RoMa for extracting dense correspondences
python main.py --config roma --images data/belv_lingua_easy --strategy bruteforce --force -V

# Triangulate dense correspondences with COLMAP
python ./scripts/dense_matching.py --sfm_dir output/"$DATASET"_superpoint+lightglue_bruteforce --dense_dir output/"$DATASET"_roma_bruteforce

