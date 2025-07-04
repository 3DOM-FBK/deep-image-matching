# FC-GNN: Recovering Reliable and Accurate Correspondences from Interferences

### [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FC-GNN_Recovering_Reliable_and_Accurate_Correspondences_from_Interferences_CVPR_2024_paper.pdf) <br/>

> FC-GNN: Recovering Reliable and Accurate Correspondences from Interferences \
> Haobo Xu, Jun Zhou, Hua Yang, Renjie Pan, Cunyan Li \
> CVPR 2024

FC-GNN is a lightweight graph neural network that optimizes local feature matching pipeline through a combined filtering and calibrating approach. It receives image pairs and related matches as inputs and returns the optimized matching results.

## Installation and demo

To install this repo:

```bash
git clone https://github.com/xuy123456/fcgnn.git && cd fcgnn
```

To use FC-GNN:

```python
from fcgnn import GNN as fcgnn
fcgnn_refiner = fcgnn()

'''
inputs:
  images: torch float tensor, gray images, normalized to [0, 1], shape: [B, 1, H, W]
  matches: torch float tensor, shape: [B, L, 4] or [L, 4]
  thd: filtering degree
  min_matches: minimum number of matches to keep

return:
  list of optimized match sets: [matches1, matches2, ...]
'''

matches_refined = fcgnn_refiner.optimize_matches(img1, img2, 
                                                 matches, 
                                                 thd=0.999,
                                                 min_matches=10)
```

We provide a script to show how to use FC-GNN with SIFT + MNN matcher:

```python
import cv2
import torch

from fcgnn import GNN as fcgnn
from sift import SIFT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sift_matcher = SIFT()
fcgnn_refiner = fcgnn().to(device)

img1_path = './assets/img1.jpg'
img2_path = './assets/img2.jpg'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

matches = sift_matcher(img1, img2, device=device)

img1_ = torch.tensor(img1.astype('float32') / 255.)[None, None].to(device)
img2_ = torch.tensor(img2.astype('float32') / 255.)[None, None].to(device)

matches_refined = fcgnn_refiner.optimize_matches(img1_, img2_, matches, thd=0.999, min_matches=10)[0]
```

The pre-trained weights will be automatically loaded. If the loading fails, you can download the weights [here](https://github.com/xuy123456/fcgnn/releases/download/v0/fcgnn.model), and put it to './weights'. You can also run demo.py to get visual results.

## BibTeX
If you find our models useful, please consider citing our paper:
```
@InProceedings{Xu_2024_CVPR,
    author    = {Xu, Haobo and Zhou, Jun and Yang, Hua and Pan, Renjie and Li, Cunyan},
    title     = {FC-GNN: Recovering Reliable and Accurate Correspondences from Interferences},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {25213-25222}
}
```
