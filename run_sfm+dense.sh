#!/bin/bash

# Set the data directory
DATA_DIR=data
DATASET=belv_lingua_easy

# Set the config and strategy
SFM_CONFIG=superpoint+lightglue
STRATEGY=bruteforce
DENSE_CONFIG=loftr

SKIP_SFM=false

INPUT_DIR=$DATA_DIR/$DATASET

# Run SfM to find camera poses to use for dense reconstruction
if [ "$SKIP_SFM" = false ] ; then
    python ./main.py --config $SFM_CONFIG --images $INPUT_DIR --quality high --tiling preselection --strategy $STRATEGY --force -V
fi

# Run dense matching (skip reconstruction with dense correspondences)
python ./main.py --config $DENSE_CONFIG --images $INPUT_DIR --quality medium --tiling preselection --strategy $STRATEGY --skip_reconstruction --force -V

# Triangulate dense correspondences with COLMAP to build a dense point cloud
python ./scripts/dense_matching.py --sfm_dir "output/${DATASET}_${SFM_CONFIG}_${STRATEGY}" --dense_dir "output/${DATASET}_${DENSE_CONFIG}_${STRATEGY}"

