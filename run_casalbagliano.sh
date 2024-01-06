#!/bin/bash

# Set the data directory
PRJ_DIR=/home/francesco/casalbagliano/subset_B

# Set the config and strategy
SFM_CONFIG=superpoint+lightglue
SFM_STRATEGY=bruteforce
SFM_QUALITY=low
SFM_TILING=none

DENSE_CONFIG=roma
DENSE_QUALITY=low
DENSE_TILING=preselection
DENSE_STRATEGY=matching_lowres

SKIP_SFM=true
DEBUG=true

if [ "$DEBUG" = true ] ; then
    VERBOSE="-V"
fi

# Run SfM to find camera poses to use for dense reconstruction
if [ "$SKIP_SFM" = false ] ; then
    python ./main.py --dir $PRJ_DIR --config $SFM_CONFIG --quality $SFM_QUALITY --tiling $SFM_TILING  --strategy $SFM_STRATEGY --force $VERBOSE
fi

# Run dense matching (skip reconstruction with dense correspondences)
python ./main.py --dir $PRJ_DIR --config $DENSE_CONFIG --quality $DENSE_QUALITY --tiling $DENSE_TILING --strategy $DENSE_STRATEGY --overlap 3 --skip_reconstruction --force $VERBOSE

# Triangulate dense correspondences with COLMAP to build a dense point cloud
SFM_DIR="$PRJ_DIR/results_${SFM_CONFIG}_${SFM_STRATEGY}_quality_${SFM_QUALITY}"
DENSE_DIR="$PRJ_DIR/results_${DENSE_CONFIG}_${DENSE_STRATEGY}_quality_${DENSE_QUALITY}"

python ./scripts/dense_matching.py --sfm_dir $SFM_DIR --dense_dir $DENSE_DIR

