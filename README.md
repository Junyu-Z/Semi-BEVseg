# Semi-Supervised Learning for Visual Bird’s Eye View Semantic Segmentation (ICRA 2024)
### [Paper](https://arxiv.org/abs/2308.14525)
*Junyu Zhu, Lina Liu, Yu Tang, Feng Wen, Wanlong Li, Yong Liu*

## Setup

### Installation & Dependency
```
git clone https://github.com/Junyu-Z/Semi-BEVseg.git
cd Semi-BEVseg
conda create -y -n bev_env python=3.7
conda activate bev_env
pip install torch==1.12.1
pip install torchvision==0.13.1
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
pip install tensorboard
pip install yacs
pip install opencv-python==4.6.0.66
pip install scikit-image
```

### Dataset
* Download the **Map expansion** and **Full dataset (v1.0)** from https://www.nuscenes.org/download. The dataset folder structure should be:
```
└── nuScenes
  ├── samples
  ├── sweeps
  ├── maps
  ├── v1.0-trainval
  ├── v1.0-test
```
* Edit the `configs/config.yml` file, setting the **nuscenes_dataroot** and **nuscenes_label_root** entries to the location of the nuScenes dataset and the desired ground truth folder respectively.

* Run the data generation script:
```
python ./scripts/generate_nuscenes_labels.py
```

* The final dataset folder structure should be:
```
└── nuScenes
  ├── samples
  ├── sweeps
  ├── maps
  ├── v1.0-trainval
  ├── v1.0-test
  ├── map-labels
```

## Training
Train  the full-sup(100% labeled data) model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 Full_Supervise.py \
--img_size 800 600 \
--tag fullSup_p1.0_600x800
```

Train the sup-only(only 2.5% labeled data) model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 Semi_Supervise_PI.py \
--img_size 800 600 \
--label_percent 0.025 \
--tag supOnly_p0.025_600x800
```

Train the proposed semi-sup(2.5% labeled data + 97.5% unlabeled data) model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 Semi_Supervise_MT.py \
--img_size 800 600 \
--label_percent 0.025 \
--tag semiSup_p0.025_600x800
```

## Acknowledgements
This project is built upon [PON](https://github.com/tom-roddick/mono-semantic-maps).

## Citation
If you find this repository useful, please cite
```bibtex
@inproceedings{zhu2024semibevseg,
  author    = {Zhu, Junyu and Liu, Lina and Tang, Yu and Wen, Feng and Li, Wanlong and Liu, Yong},
  title     = {Semi-Supervised Learning for Visual Bird’s Eye View Semantic Segmentation}, 
  booktitle = {ICRA},
  year      = {2024},
}
```
