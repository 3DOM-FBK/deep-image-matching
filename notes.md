# To Do List

## Priority To Do

- [ ] Improve speed, expecially for large datasets
- [ ] Store matches at low resolution and use them in all the following steps when they are needed (without extracting them again)
- [ ] Prepare pycolmap.yaml and calibration.yaml (for both colmap and openmvg)
- [ ] Testing on very large datasets ([Issue [#29](https://github.com/3DOM-FBK/deep-image-matching/issues/29)])
- [ ] Use Github submodules instead of copying thirdpary code inside the repo
- [ ] Add subpixel refinement of the matches (e.g., cross-correlation or [pixel-perfect-sfm](https://github.com/cvg/pixel-perfect-sfm))

## Bugs and Issues

- [ ] seems there is a bug in kornia when matching more than 1000 images
- [ ] GUI is broken with the new configuration management
- [ ] SE2-LoFTR not fully implemented
- [ ] DeDoDe and RoMa are not working without CUDA
- [ ] open issue on pydegensac for pairs with many tie points (>10000). Quite random results with LoFTR and RoMa
- [ ] script run from scripts folder are not able to import modules!
- [ ] scripts/delete_static_matches.py not ready
- [ ] Avoid importing duplicated matches into colmap database when working with tiles bug (see Issue [[#8](https://github.com/3DOM-FBK/deep-image-matching/issues/8)])

## Other enhancements

- [ ] Improve configuration management [_Hydra_](https://hydra.cc/docs/tutorials/structured_config/schema/) to make using yaml files, command line and GUI (Issue [[#48](https://github.com/3DOM-FBK/deep-image-matching/issues/48)])
- [ ] Add steerers + DeDoDe
- [ ] Add Silk features
- [ ] Add SIFT + LightGlue
- [ ] Support for more hand-crafted local features from kornia and opencv (e.g., affine SIFT, AKAZE, etc.)
- [ ] Extend documentation
- [ ] Workflow to rebuild & publish image to Docker Hub
- [ ] Integrate support for Pix4D [Open Photogrammetry Format](https://github.com/Pix4D/opf-spec)
- [ ] Improve GUI (e.g., [image-matching-webui](https://github.com/Vincentqyw/image-matching-webui/tree/main))
- [ ] Improve visualization routines for extracted features and matches
- [ ] Extend compatibility to other software (e.g., Meshroom)
- [ ] Remove unmactched features from the h5 file and from COLMAP database after the matching is completed enhancement (Issue [[#30](https://github.com/3DOM-FBK/deep-image-matching/issues/30)])
- [ ] Support for 16-bit images (Issue [[#42](https://github.com/3DOM-FBK/deep-image-matching/issues/42)])
- [ ] Improve Tiler with adaptive tile grid computation enhancement (Issue [[#23](https://github.com/3DOM-FBK/deep-image-matching/issues/23)])
- [ ] keep matching score/confidence (if available) (Issue [[#22](https://github.com/3DOM-FBK/deep-image-matching/issues/22)])
- [ ] Allow to read images from different folders (e.g., subfolders)
- [ ] Support for exporting matches in COLMAP database with different camera calibration based on the input images (e.g., on exif or on subfolder structure) (Issue [[#45](https://github.com/3DOM-FBK/deep-image-matching/issues/45)])
- [ ] Apply masks during feature extraction
- [ ] Automatically download weights for all the models
- [ ] Autoselect tiling grid in order to fit images in GPU memory

## Completed

- [x] Tile processing for high resolution images
- [x] Manage image rotations
- [x] Add image retrieval with global descriptors
- [x] add GUI
- [x] Add pycolmap compatibility
- [x] Add exporting to Bundler format ready for importing into Metashape (only on Linux and MacOS by using pycolmap)
- [x] Dockerization
- [x] Add tests, documentation and examples (e.g. colab, ..)
- [x] Cleanup repository to removed large files from Git history
- [x] Update README CLI options
- [x] Make semi-dense matcher work with multi-camera (Issue [[#24](https://github.com/3DOM-FBK/deep-image-matching/issues/24)])
- [x] Improve usage of multiple descriptors together
- [x] Finish extending compatibility to OpenMVG
- [x] Tests on satellite images
