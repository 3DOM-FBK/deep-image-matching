#!/bin/bash

# Set the data directory
ROOT_DIR=/home/francesco/casalbagliano/subset_A

DATASET=casalbagliano

# Set the config and strategy
SFM_CONFIG=superpoint+lightglue
SFM_STRATEGY=bruteforce
DENSE_CONFIG=roma
DENSE_STRATEGY=sequential

SKIP_SFM=false
DEBUG=true

IMAGE_DIR=$ROOT_DIR/images
OUTPUT_SFM_DIR="$ROOT_DIR/${DATASET}_${SFM_CONFIG}_${SFM_STRATEGY}"
OUTPUT_DENSE_DIR="$ROOT_DIR/${DATASET}_${DENSE_CONFIG}_$DENSE_STRATEGY"

if [ "$DEBUG" = true ] ; then
    db_key="-V"
fi

# Run SfM to find camera poses to use for dense reconstruction
if [ "$SKIP_SFM" = false ] ; then
    python ./main.py --config $SFM_CONFIG --images $IMAGE_DIR --outs $OUTPUT_SFM_DIR --quality high --tiling preselection --strategy $SFM_STRATEGY --force $db_key
fi

# Run dense matching (skip reconstruction with dense correspondences)
python ./main.py --config $DENSE_CONFIG --images $IMAGE_DIR --quality low --tiling preselection --strategy $DENSE_STRATEGY --overlap 2 --skip_reconstruction --force $db_key

# Triangulate dense correspondences with COLMAP to build a dense point cloud
python ./scripts/dense_matching.py --sfm_dir $OUTPUT_SFM_DIR --dense_dir $OUTPUT_DENSE_DIR

