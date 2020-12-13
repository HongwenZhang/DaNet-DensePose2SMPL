# Learning 3D Human Shape and Pose from Dense Body Parts

This repository includes the PyTorch code of the network described in [Learning 3D Human Shape and Pose from Dense Body Parts](https://hongwenzhang.github.io/dense2mesh/pdf/learning3Dhuman.pdf).

[![Project Page](https://hongwenzhang.github.io/dense2mesh/img/framework.png "Project Page")](https://hongwenzhang.github.io/dense2mesh)

## Requirements

- Python 3.6.10

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.1.0

- [Neural Renderer](https://github.com/daniilidis-group/neural_renderer)

- [opendr](https://gitlab.eecs.umich.edu/ngv-python-modules/opendr#) optional

- other packages listed in `requirements.txt`

### necessary files

> DensePose UV data

- Run the following script to fetch DensePose UV data.

```
bash get_densepose_uv.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files and put them into the `./data/smpl` directory.

> Fetch preprocessed data from [SPIN](https://github.com/nkolot/SPIN#fetch-data).

> Download the [pre-trained models](https://drive.google.com/drive/folders/1vP3HxsMHdB3_2lthLDq1RTsVeBnOJlpC?usp=sharing) and put them into the `./data/pretrained_model` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── dataset_extras
│   └── .npz files
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── pretrained_model
│   ├── .pt files
│   └── learned_ratio.pkl
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
└── UV_data
    ├── UV_Processed.mat
    └── UV_symmetry_transforms.mat
```


## Demo

1. Run the demo code. Using `--use_opendr` if the `opendr` package is successfully installed.

```
python3 demo.py --checkpoint=data/pretrained_model/danet_model_h36m_itw.pt --img_dir ./examples --use_opendr
```

2. View visualization results in `./output`. Results are organized (from left to right) as the input image, the estimated IUV maps (global and partial), the rendered IUV of the predicted SMPL model, the predicted SMPL model (front and side views).

<p align='center'>
<img src='https://hongwenzhang.github.io/dense2mesh/img/demo_result.png' title='demo results' style='max-width:600px'></img>
</p>

## Evaluation

### Human3.6M / 3DPW

Run the evaluation code. Using `--dataset` to specify the evaluation dataset.
```
# Example usage:

# Human3.6M Protocol 2
python3 eval.py --checkpoint=data/pretrained_model/danet_model_h36m_itw.pt --dataset=h36m-p2 --log_freq=20

# 3DPW
python3 eval.py --checkpoint=data/pretrained_model/danet_model_h36m_itw.pt --dataset=3dpw --log_freq=20
```

### COCO Keypoint Localization

1. Download the preprocessed data [coco_2014_val.npz](https://drive.google.com/drive/folders/1vP3HxsMHdB3_2lthLDq1RTsVeBnOJlpC?usp=sharing). Put it into the `./data/dataset_extras` directory. 

2. Run the COCO evaluation code.
```
python3 eval_coco.py --checkpoint=data/pretrained_model/danet_model_h36m_dpcoco.pt
```

## Training

TODO

## Citation
If this work is helpful in your research, please cite the following paper.
```
@article{zhang2020learning,
  title={Learning 3D Human Shape and Pose from Dense Body Parts},
  author={Zhang, Hongwen and Cao, Jie and Lu, Guo and Ouyang, Wanli and Sun, Zhenan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020}
}
```

## Acknowledgments

The code is developed upon the following projects. Many thanks to the original authors.

- [SPIN](https://github.com/nkolot/SPIN)

- [pytorch_HMR](https://github.com/MandyMo/pytorch_HMR)

- [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

- [pose_resnet](https://github.com/Microsoft/human-pose-estimation.pytorch)

- [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch)

- [DensePose](https://github.com/facebookresearch/DensePose)
