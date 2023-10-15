# deep-image-matching
Install hloc. See https://github.com/cvg/Hierarchical-Localization

# Example usage
Before running check options with `python ./main.py --help`, then:
```
python ./main.py -i assets/imgs -o assets/outs -m sequential -f superglue -n 8000 -v 1
```
See other examples in run.bat. If you want customize detector and descpritor options, change default options in config.py. 


# TODO
- [X] added kornia features
- [X] multiview works for TileSelection.NONE
- [ ] bugs on tile processing
- [X] move configuration params superglue and loftr in config.py
- [ ] organize repo as Francesco repo
- [ ] repo works per pairs - avoid extract again keypoints on already processed images
- [ ] add image retrieval with global descriptors
- [ ] geometric verification with degensac
- [ ] add GUI
- [ ] KeyNetAffNetHardNet seems too work only on very small images, probably a BUG (error GPU memory)
- [ ] lightglue with DISK features to be implemented
- [ ] self.max_feat_numb: To be checked that works on all detectors and matchers (probably no). See image_matching.py