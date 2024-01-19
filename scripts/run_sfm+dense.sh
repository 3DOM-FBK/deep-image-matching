#!/bin/bash

# Set the data directory
# DATA_DIR=data
# DATASET=belv_lingua_easy
DATA_DIR=assets
DATASET=example_cyprus

# Set the config and strategy
SFM_CONFIG=superpoint+lightglue
STRATEGY=bruteforce
DENSE_CONFIG=loftr

SKIP_SFM=true
DEBUG=true

INPUT_DIR=$DATA_DIR/$DATASET
if [ "$DEBUG" = true ] ; then
    db_key="-V"
fi

# Run SfM to find camera poses to use for dense reconstruction
if [ "$SKIP_SFM" = false ] ; then
    python ./main.py --config $SFM_CONFIG --images $INPUT_DIR --quality high --tiling preselection --strategy $STRATEGY --force $db_key --upright
fi

# Run dense matching (skip reconstruction with dense correspondences)
python ./main.py --config $DENSE_CONFIG --images $INPUT_DIR --quality medium --tiling preselection --strategy sequential --overlap 3 --skip_reconstruction --force $db_key --upright

# Triangulate dense correspondences with COLMAP to build a dense point cloud
python ./scripts/dense_matching.py --sfm_dir "output/${DATASET}_${SFM_CONFIG}_${STRATEGY}" --dense_dir "output/${DATASET}_${DENSE_CONFIG}_${STRATEGY}"

