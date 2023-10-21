# Example usage
Before running check options with `python ./main.py --help`, then:
```
python ./main.py -i assets/imgs -o assets/outs -m sequential -f superglue -n 8000 -v 1
```
See other examples in run.bat. If you want customize detector and descpritor options, change default options in config.py. 


# Multiview tests
- [X] SuperGlue
- [X] LightGlue
- [ ] LoFTR -- PROBLEM -- LoFTR to be rewritten with code from Mishkin for multivew!!!!!!!!! URGENT !!!!!!!!!!!!!!
- [X] ALIKE
- [X] ORB -- CHECK -- performance very bad matching nadiral and oblique images
- [X] DISK
- [X] Superpoint
- [X] KeyNetHardNet


# TODO
- [X] added kornia features
- [X] multiview works for TileSelection.NONE
- [ ] extend to tile processing
- [X] move configuration params superglue and loftr in config.py
- [ ] repo works per pairs - avoid extract again keypoints on already processed images
- [ ] add image retrieval with global descriptors
- [X] add GUI
- [ ] lightglue with DISK features to be implemented
- [ ] self.max_feat_numb: To be checked that works on all detectors and matchers (probably no). See image_matching.py. For alike do not work
- [ ] manage rotation, while majority of local features do not deal with them
- [ ] add ALIKED
- [ ] add SIFT from opencv
- [ ] look for open source SuperPoint