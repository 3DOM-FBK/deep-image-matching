from pathlib import Path

import deep_image_matching as dim
import h5py

params = {
    "dir": "./assets/example_cyprus",
    "pipeline": "superpoint+lightglue",
    "strategy": "bruteforce",
    "quality": "medium",
    "tiling": "none",
    "camera_options": "./assets/example_cyprus/cameras.yaml",
    "openmvg": None,
}
config = dim.Config(params)

# Get image list
img_list = list((Path(params["dir"]) / "images").rglob("*"))

# Manually load the desired extractor
extractor = dim.extractors.SuperPointExtractor(config)

# Extract features from the first image
feat_path = extractor.extract(img_list[0])

# Open the feature file and print its contents
with h5py.File(feat_path, "r") as f:
    for k, v in f.items():
        print(k, v)

print("Done")
