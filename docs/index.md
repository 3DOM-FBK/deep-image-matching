# DEEP-IMAGE-MATCHING

| SIFT                                               | DISK                                                 | IMAGES ORIENTATION                                     | DENSE WITH ROMA                                   |
| -------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------- |
| <img src='./assets/matches_sift.gif' height="100"> | <img src='./assets/matches_joined.gif' height="100"> | <img src='./assets/orientation_deep.gif' height="100"> | <img src='./assets/roma_dense.gif' height="100"> |

| SIFT                                               | SUPERGLUE                                              |
| -------------------------------------------------- | ------------------------------------------------------ |
| <img src='./assets/temple_rsift.gif' height="165"> | <img src='./assets/temple_superglue.gif' height="165"> |

Multivew matcher for SfM software. Support both deep-learning based and hand-crafted local features and matchers and export keypoints and matches directly in a COLMAP database or to Agisoft Metashape by importing the reconstruction in Bundler format. Now, it supports both OpenMVG and MicMac. Feel free to collaborate!

While `dev` branch is more frequently updated, `master` is the default more stable branch and is updated from `dev` less frequently. If you are looking for the newest developments, please switch to `dev`.

**Please, note that `deep-image-matching` is under active development** and it is still in an experimental stage. If you find any bug, please open an issue.

Key features:

- Multiview
- Large format images
- SOTA deep-learning and hand-crafted features
- Support for image rotations
- Compatibility with several SfM software
- Support image retrieval with deep-learning local features

| Supported Extractors               | Supported Matchers                                        |
| ---------------------------------- | --------------------------------------------------------- |
| &check; SuperPoint                 | &check; Lightglue (with Superpoint, Disk, and ALIKED)     |
| &check; DISK                       | &check; SuperGlue (with Superpoint)                       |
| &check; ALIKE                      | &check; LoFTR                                             |
| &check; ALIKED                     | &#x2610; SE2-LoFTR                                        |
| &#x2610; Superpoint free           | &check; Nearest neighbor (with KORNIA Descriptor Matcher) |
| &check; KeyNet + OriNet + HardNet8 | &check; RoMa                                              |
| &check; ORB (opencv)               | &#x2610; GlueStick                                        |
| &check; SIFT (opencv)              |
| &check; DeDoDe                     |

| Supported SfM software                        |
| --------------------------------------------- |
| &check; COLMAP                                |
| &check; OpenMVG                               |
| &check; Agisoft Metashape                     |
| &check; Software that supports bundler format |
