[![Static Badge](https://img.shields.io/badge/Powered_by-Kornia-green)](https://github.com/kornia/kornia) [![Static Badge](https://img.shields.io/badge/Matches_for-COLMAP-red)](https://github.com/colmap/colmap)

# DEEP-IMAGE-MATCHING
Multivew matcher for COLMAP. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database.
The repo is under construction but it already works with SuperGlue, LightGlue, ALIKE, DISK, Key.Net+HardNet8, ORB.
Feel free to collaborate!

# Example usage
Before running check options with `python ./main.py --help`, then:
```
python ./main.py -i assets/imgs -o assets/outs -m sequential -f superglue -n 8000 -v 1
```
See other examples in run.bat. If you want to customize detector and descpritor options, change default options in config.py. 

# Multiview tests
- [X] SuperGlue
- [X] LightGlue with SuperGlue
- [ ] LoFTR
- [X] ALIKE
- [X] ORB opencv
- [X] DISK
- [X] Superpoint
- [ ] Superpoint free
- [X] KeyNet + OriNet + HardNet8
- [ ] SIFT opencv
- [ ] ALIKED

# TODO
- [X] add kornia features
- [ ] extend to tile processing
- [ ] add image retrieval with global descriptors
- [X] add GUI
- [ ] lightglue with DISK features to be implemented
- [ ] manage rotation