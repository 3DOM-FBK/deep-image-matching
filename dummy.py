from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
)

images = Path("data/hard_lowres")

outputs = Path("res/sfm/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["superglue"]

# ## Find image pairs via image retrieval
# We extract global descriptors with NetVLAD and find for each image the most similar ones. For smaller dataset we can instead use exhaustive matching via `hloc/pairs_from_exhaustive.py`, which would find $\frac{n(n-1)}{2}$ images pairs.

# retrieval_path = extract_features.main(retrieval_conf, images, outputs)
# pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)

# ## Extract and match local features

feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

# ## 3D reconstruction
# Run COLMAP on the features and matches.

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

# ## Visualization
# We visualize some of the registered images, and color their keypoint by visibility, track length, or triangulated depth.

visualization.visualize_sfm_2d(model, images, color_by="visibility", n=5)

visualization.visualize_sfm_2d(model, images, color_by="track_length", n=5)

visualization.visualize_sfm_2d(model, images, color_by="depth", n=5)
