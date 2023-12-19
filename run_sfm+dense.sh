#!/bin/bash

# Set the data directory
DATA_DIR=data
DATASET=belv_lingua_easy

# Set the config and strategy
SFM_CONFIG=superpoint+lightglue
STRATEGY=bruteforce
DENSE_CONFIG=loftr

SKIP_SFM=true

INPUT_DIR=$DATA_DIR/$DATASET

# Run SfM with superpoint+superglue
if [ "$SKIP_SFM" = false ] ; then
    python ./main.py --config $SFM_CONFIG --images $INPUT_DIR --strategy $STRATEGY --force -V
fi

# # Run RoMa for extracting dense correspondences
python ./main.py --config $DENSE_CONFIG --images $INPUT_DIR --strategy $STRATEGY --force -V

# Triangulate dense correspondences with COLMAP
python ./scripts/dense_matching.py --sfm_dir "output/${DATASET}_${SFM_CONFIG}_${STRATEGY}" --dense_dir "output/${DATASET}_${DENSE_CONFIG}_${STRATEGY}"

