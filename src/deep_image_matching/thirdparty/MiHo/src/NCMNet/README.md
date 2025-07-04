# NCMNet
(CVPR 2023) PyTorch implementation of Paper "Progressive Neighbor Consistency Mining for Correspondence Pruning"

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.


# Citing NCMNet
If you find the NCMNet code useful, please consider citing:

```bibtex
@inproceedings{liu2023ncmnet,
  author    = {Liu, Xin and Yang, Jufeng},
  title     = {Progressive Neighbor Consistency Mining for Correspondence Pruning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {9527-9537}
}
```

# Preparing Data
Please follow their instructions to download the training and testing data.
```bash
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8 ## YFCC100M
tar -xvf raw_data_yfcc.tar.gz

bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2 ## SUN3D
tar -xvf raw_sun3d_test.tar.gz
bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
tar -xvf raw_sun3d_train.tar.gz
```
 
After downloading the datasets, the initial matches for YFCC100M and SUN3D can be generated as following. Here we provide descriptors for SIFT (default), ORB, and SuperPoint.
```bash
cd dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```

# Testing and Training Model
We provide a pretrained model on YFCC100M. The results in our paper can be reproduced by running the test script:
```bash
cd code 
python main.py --run_mode=test --model_path=../model/yfcc --res_path=../model/yfcc 
```
Set `--use_ransac=True` to get results after RANSAC post-processing.

If you want to retrain the model on YFCC100M, run the tranining script.
```bash
cd code 
python main.py 
```

You can also retrain the model on SUN3D by modifying related settings in `code\config.py`.

# Acknowledgement
This code is heavily borrowed from [[OANet](https://github.com/zjhthu/OANet)] [[CLNet](https://github.com/sailor-z/CLNet)]. If you use the part of code related to data generation, testing, or evaluation, you should cite these papers:
```bibtex
@inproceedings{zhang2019oanet,
  title={Learning Two-View Correspondences and Geometry Using Order-Aware Network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2019}
}
@inproceedings{zhao2021clnet,
  title={Progressive Correspondence Pruning by Consensus Learning},
  author={Zhao, Chen and Ge, Yixiao and Zhu, Feng and Zhao, Rui and Li, Hongsheng and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```  
