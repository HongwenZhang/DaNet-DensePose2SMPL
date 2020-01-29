# Learning 3D Human Shape and Pose from Dense Body Parts

This repository includes the PyTorch code of the network described in [Learning 3D Human Shape and Pose from Dense Body Parts](https://hongwenzhang.github.io/dense2mesh/pdf/learning3Dhuman.pdf).

[![Project Page](https://hongwenzhang.github.io/dense2mesh/img/framework.png "Project Page")](https://hongwenzhang.github.io/dense2mesh)

## Requirements

- python 2.7

### packages

- [PyTorch](https://www.pytorch.org) tested on version 1.0.1.post2

- [Neural Renderer](https://github.com/daniilidis-group/neural_renderer)

- [opendr](https://github.com/mattloper/opendr) optional

- [smpl_webuser](https://smpl.is.tue.mpg.de) optional

- other packages listed in `requirements.txt`

### necessary files

> DensePose UV data

- Run the following script to fetch DensePose UV data.

```
bash get_densepose_uv.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Put model files into the `./data/SMPL_data` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── UV_data
│   ├── UV_Processed.mat
│   └── UV_symmetry_transforms.mat
└── SMPL_data
    ├── basicModel_f_lbs_10_207_0_v1.0.0.pkl
    ├── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
    └── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

### Demo
1. Download the [pre-trained model](https://drive.google.com/drive/folders/1XlclE5EEX6OPWtQ9p1oubBDAqOcdhuqD?usp=sharing) (trained on [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [DensePose-COCO](https://densepose.org)) and put it into the `./data/pretrained_model` directory.

2. Run the demo code. Using `--use_opendr` if the `opendr` package is successfully installed.

```
python demo.py  --cfg configs/danet_demo.yaml --load_ckpt ./data/pretrained_model/danet_model_h36m_cocodp.pth --img_dir ./imgs --use_opendr
```

3. View visualization results in `./output`. Results are organized (from left to right) as the input image, the estimated IUV maps (global and partial), the rendered IUV of the predicted SMPL model, the predicted SMPL model (front and side views).

<p align='center'>
<img src='https://hongwenzhang.github.io/dense2mesh/img/demo_result.png' title='demo results' style='max-width:600px'></img>
</p>

## Citation
If this work is helpful in your research, please cite the following papers.
```
@inproceedings{zhang2019danet,
  title={DaNet: Decompose-and-aggregate Network for 3D Human Shape and Pose Estimation},
  author={Zhang, Hongwen and Cao, Jie and Lu, Guo and Ouyang, Wanli and Sun, Zhenan},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={935--944},
  year={2019},
  organization={ACM}
}

@article{zhang2019learning,
  title={Learning 3D Human Shape and Pose from Dense Body Parts},
  author={Zhang, Hongwen and Cao, Jie and Lu, Guo and Ouyang, Wanli and Sun, Zhenan},
  journal={arXiv preprint arXiv:1912.13344},
  year={2019}
}
```

## Acknowledgments

The code is developed upon the following projects. Thanks to the original authors.

- [pytorch_HMR](https://github.com/MandyMo/pytorch_HMR)

- [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

- [pose_resnet](https://github.com/Microsoft/human-pose-estimation.pytorch)

- [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch)

- [DensePose](https://github.com/facebookresearch/DensePose)

