# ACNe: Attentive Context Normalization for Robust Permutation-Equivariant Learning (CVPR2020)

## Introduction
This repository is the Tensorflow implementation for "ACNe: Attentive Context Normalization for Robust Permutation-Equivariant Learning" by [Weiwei Sun](https://weiweisun2018.github.io/), [Wei Jiang](https://jiangwei221.github.io/), [Eduard Trulls](http://etrulls.github.io/), [Andrea Tagliasacchi](http://gfx.uvic.ca/people/ataiya), [Kwang Moo Yi](http://vision.uvic.ca/people/kmyi). If you use this code in your research, please cite the paper. In addition, if you are using the part of the code related to CNe, please also cite the CNe paper.

[[Paper](https://arxiv.org/abs/1907.02545)] &emsp; [[code](https://github.com/vcg-uvic/acne)]

### Citations
        @inproceedings{sun2020acne,
          title={Attentive Context Normalization for Robust Permutation-Equivariant Learning},
          author={Weiwei Sun, Wei Jiang, Andrea Tagliasacchi, Eduard Trulls, Kwang Moo Yi},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2020},
        }
        @inproceedings{yi2018learning,
          title={Learning to Find Good Correspondences},
          author={Kwang Moo Yi* and Eduard Trulls* and Yuki Ono and Vincent Lepetit and Mathieu Salzmann and Pascal Fua},
          booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
          year={2018}
        }

### Abstraction
![teaser](docs/teaser.png)
Many problems in computer vision require dealing with sparse, unordered data in the form of point clouds. Permutation-equivariant networks have become a popular solution-they operate on individual data points with simple perceptrons and extract contextual information with global pooling. This can be achieved with a simple normalization of the feature maps, a global operation that is unaffected by the order. In this paper, we propose Attentive Context Normalization (ACN), a simple yet effective technique to build permutation-equivariant networks robust to outliers. Specifically, we show how to normalize the feature maps with weights that are estimated within the network, excluding outliers from this normalization. We use this mechanism to leverage two types of attention: local and global-by combining them, our method is able to find the essential data points in high-dimensional space to solve a given task. We demonstrate through extensive experiments that our approach, which we call Attentive Context Networks (ACNe), provides a significant leap in performance compared to the state-of-the-art on camera pose estimation, robust fitting, and point cloud classification under noise and outliers.


## Requirements
Please use Tensorflow (1.8 is recommended). Please also install other dependencies via `pip`

## Usage 

### Data Preparation
We used the dataset from CNe and OANet. Note that CNe dataset has a very small size. Users can quickly run the code with this dataset.

#### Dataset from CNe
We use the following script to generate data for `st_peters_square`. It will create a folder `data_dump/st_peters_square`. 
```
mkdir datasets
cd datasets
wget http://webhome.cs.uvic.ca/~kyi/files/2018/learned-correspondence/st_peters_square.tar.gz
tar -xvf st_peters_square.tar.gz
rm st_peters_square.tar.gz
cd ..
./dump_data.py
```

#### Dataset from OANet 
We use OANet's [official repo](https://github.com/zjhthu/OANet.git) to generate the dataset. It will create a folder `data_dump`. Please link the data folder as `data_dump_oan`.

### Training:

Fundamental Matrix Estimation
```
bash jobs/acne.sh train
```

Essential Matrix Estimation
```
bash jobs/acne_E.sh train
```

For easy comparison, scripts for CNe (i.e., `jobs/CNe.sh`) are also provided.

### Testing:
We provided the [pretrained model here](https://drive.google.com/open?id=13VREoILjghalQYQO_35nQIqrhs_zbajw).

Fundamental Matrix Estimation
```
bash jobs/acne.sh
```

Essential Matrix Estimation
```
bash jobs/acne_E.sh
```

Please note that, in the testing phase, we use the bidirectional check by setting `--prefiltering=B`. It can be disabled by removing this argument.

## Acknowledgement
The code is heavily based on [CNe](https://github.com/vcg-uvic/learned-correspondence-release).

The code for fundamental matrix estimation is partly borrowed from [OANet](https://github.com/zjhthu/OANet.git). If you use the part of code related to fundamental matrix estimation, please also cite OANet paper:
```
@article{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  journal={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```
