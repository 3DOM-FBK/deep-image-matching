# OANet implementation

Pytorch implementation of OANet for ICCV'19 paper ["Learning Two-View Correspondences and Geometry Using Order-Aware Network"](https://arxiv.org/abs/1908.04964), by Jiahui Zhang, Dawei Sun, Zixin Luo, Anbang Yao, Lei Zhou, Tianwei Shen, Yurong Chen, Long Quan and Hongen Liao.

This paper focuses on establishing correspondences between two images. We introduce the DiffPool and DiffUnpool layers to capture the local context of unordered sparse correspondences in a learnable manner. By the collaborative use of DiffPool operator, we propose Order-Aware Filtering block which exploits the complex global context.

This repo contains the code and data for essential matrix estimation described in our ICCV paper. Besides, we also provide code for fundamental matrix estimation and the usage of side information (ratio test and mutual nearest neighbor check). Documents about this part will also be released soon.

Welcome bugs and issues!

If you find this project useful, please cite:

```
@article{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  journal={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.


## Example scripts
## Note: Since 2023.04, all the data and model will be stored on [google drive](https://drive.google.com/drive/folders/1qbnrDSXJhnXC0MiZNNqW7vezAHlkBNgo?usp=sharing), previous link to Altizure platform will not be available. Please download corresponding data from google drive.
### Run the demo

For a quick start, clone the repo and download the pretrained model.
```bash
git clone https://github.com/zjhthu/OANet.git 
cd OANet 
wget https://research.altizure.com/data/oanet_data/model_v2.tar.gz 
tar -xvf model_v2.tar.gz
cd model
wget https://research.altizure.com/data/oanet_data/sift-gl3d.tar.gz
tar -xvf sift-gl3d.tar.gz
```

Then run the fundamental matrix estimation demo.

```bash
cd ./demo && python demo.py
```

### Generate training and testing data

First download YFCC100M dataset.
```bash
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
tar -xvf raw_data_yfcc.tar.gz
```

Download SUN3D testing (1.1G) and training (31G) dataset if you need.
```bash
bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```

Then generate matches for YFCC100M and SUN3D (only testing). Here we provide scripts for SIFT, this will take a while.
```bash
cd dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```
Generate SUN3D training data if you need by following the same procedure and uncommenting corresponding lines in `sun3d.py`.



### Test pretrained model

We provide the model trained on YFCC100M and SUN3D described in our ICCV paper. Run the test script to get results in our paper.


```bash
cd ./core 
python main.py --run_mode=test --model_path=../model/yfcc/essential/sift-2000 --res_path=../model/yfcc/essential/sift-2000/ --use_ransac=False
python main.py --run_mode=test --data_te=../data_dump/sun3d-sift-2000-test.hdf5 --model_path=../model/sun3d/essential/sift-2000 --res_path=../model/sun3d/essential/sift-2000/ --use_ransac=False
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

### Train model on YFCC100M

After generating dataset for YFCC100M, run the tranining script.
```bash
cd ./core 
python main.py
```

You can train the fundamental estimation model by setting `--use_fundamental=True --geo_loss_margin=0.03` and use side information by setting `--use_ratio=2 --use_mutual=2`

### Train with your own local feature or data 

The provided models are trained using SIFT. You had better retrain the model if you want to use OANet with 
your own local feature, such as ContextDesc, SuperPoint and etc. 

You can follow the provided example scirpts in `./dump_match` to generate dataset for your own local feature or data.

Tips for training OANet: if your dataset is small and overfitting is observed, you can consider replacing the `OAFilter` with `OAFilterBottleneck`.

Here we also provide a pretrained essential matrix estimation model using ContextDesc on YFCC100M.
```bash
cd model/
wget https://research.altizure.com/data/oanet_data/contextdesc-yfcc.tar.gz
tar -xvf contextdesc-yfcc.tar.gz
```
To test this model, you need to generate your own data using ContextDesc and then run `python main.py --run_mode=test --data_te=YOUR/OWN/CONTEXTDESC/DATA --model_path=../model/yfcc/essential/contextdesc-2000 --res_path=XX --use_ratio=2`.

## Application on 3D reconstructions

<p><img src="https://github.com/zjhthu/OANet/blob/master/media/sfm.png" alt="sample" width="70%"></p>

<!-- Reconstructions from the Alamo Dataset. -->

## News

1. Together with the local feature [ContextDesc](https://github.com/lzx551402/contextdesc), we won both the stereo and muti-view tracks at the [CVPR19 Image Matching Challenge](https://image-matching-workshop.github.io/leaderboard/) (June. 2, 2019).

2. We also rank the third place on the [Visual Localization Benchmark](https://www.visuallocalization.net/workshop/cvpr/2019/) using ContextDesc (Aug. 30, 2019).

## Acknowledgement
This code is heavily borrowed from [Learned-Correspondence](https://github.com/vcg-uvic/learned-correspondence-release). If you use the part of code related to data generation, testing and evaluation, you should cite this paper and follow its [license](https://github.com/zjhthu/OANet/blob/master/LICENSES/LICENSE_UVIC_EPFL).

```
@inproceedings{yi2018learning,
  title={Learning to Find Good Correspondences},
  author={Kwang Moo Yi* and Eduard Trulls* and Yuki Ono and Vincent Lepetit and Mathieu Salzmann and Pascal Fua},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

## Changelog

### 2019.09.29 
* Release code for data generation.
### 2019.10.04
* Release model and data for SUN3D.
### 2019.12.09
* Release a general purpose model trained on [GL3D-v2](https://github.com/lzx551402/GL3D/tree/v2), which has been tested on [FM-Benchmark](https://github.com/JiawangBian/FM-Bench). This model achieves 66.1/92.3/84.0/47.0 on TUM/KITTI/T&T/CPC respectively using SIFT.
* Release model trained using ContextDesc.

