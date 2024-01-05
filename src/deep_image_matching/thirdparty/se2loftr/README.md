# SE2-LoFTR
This is the repo for the CVPR Image Matching Workshop paper [*A case for using rotation invariant features in state of the art feature matchers*](https://arxiv.org/abs/2204.10144).
We implement a rotation equivariant LoFTR-version by using steerable CNNs.

Please see the [LoFTR repo](https://github.com/zju3dv/LoFTR) or the file `LoFTR_README.md` for instructions on how to obtain the data and run the code.
We add a single dependency, namely [e2cnn](https://github.com/QUVA-Lab/e2cnn).

The new config-files `configs/loftr/outdoor/loftr_ds_e2_dense*.py` contain the parameters used for our SE2-LoFTR experiments.

Models trained on MegaDepth can be found [here](https://drive.google.com/drive/folders/1Wiq5wlrg2rhope5Xd_MIKckAnjrjlh1a).

## TODOs
* Implement the rotation equivariant backbone as an `EquivariantModule`.

## Cite
If you find this code useful in your research, please cite our paper as well as the LoFTR and e2cnn papers:

```
@inproceedings{bokman2022se2loftr,
    title={A case for using rotation invariant features in state of the art feature matchers},
    author={B\"okman, Georg and Kahl, Fredrik},
    booktitle={CVPRW},
    year={2022}
}

@article{sun2021loftr,
    title={{LoFTR}: Detector-Free Local Feature Matching with Transformers},
    author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
    journal={CVPR},
    year={2021}
}

@inproceedings{e2cnn,
    title={{General E(2)-Equivariant Steerable CNNs}},
    author={Weiler, Maurice and Cesa, Gabriele},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2019},
}
```
