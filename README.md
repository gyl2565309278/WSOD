# WSOD

Some basic methods of weakly supervised object detection (WSOD), containing methods such as [WSDDN](https://openaccess.thecvf.com/content_cvpr_2016/html/Bilen_Weakly_Supervised_Deep_CVPR_2016_paper.html), [OICR](https://openaccess.thecvf.com/content_cvpr_2017/html/Tang_Multiple_Instance_Detection_CVPR_2017_paper.html), [PCL](https://ieeexplore.ieee.org/document/8493315) and so on.

This project is based on [Detectron2](https://github.com/facebookresearch/detectron2).

## Installation

First, you should build Detectron2 v0.6 from source. See [Detectron2 v0.6 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html). Then you will have the following directory structure:

```
detectron2
|_ configs
|_ datasets
|_ detectron2
|_ projects
|  |_ DeepLab
|  |_ DensePose
|  |_ MViTv2
|  |_ ...
|_ ...
```

Next, install and build this project as follows:

```
cd detectron2/projects

python -m pip install 'git+https://github.com/gyl2565309278/WSOD.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/gyl2565309278/WSOD.git
python -m pip install -e WSOD

cd ../..
```

At last, you should create `models/` and `output/` folders under detectron2 directory.

```
mkdir detectron2/models detectron2/output

# Or, use soft link to build the two folders:
ln -s /path/to/models detectron2/models
ln -s /path/to/output detectron2/output
```

## Datasets Preparation

Please follow [README.md](https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md) under `detectron2/datasets/` to build datasets.

In addition, you should download MCG proposals from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg) to `detectron2/datasets/proposals/`, and transform them to pickle serialization format:

```
cd detectron2

mkdir datasets/proposals

# Or, use soft link to build the folder:
ln -s /path/to/proposals datasets/proposals

# Pascal VOC 2007
python projects/WSOD/tools/convert_proposals.py voc_2007_train datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/voc_2007_train_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py voc_2007_val datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/voc_2007_val_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py voc_2007_trainval datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/voc_2007_trainval_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py voc_2007_test datasets/proposals/MCG-Pascal-Main_trainvaltest_2007-boxes datasets/proposals/voc_2007_test_mcg_proposals_d2.pkl

# Pascal VOC 2012
python projects/WSOD/tools/convert_proposals.py voc_2012_train datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/voc_2012_train_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py voc_2012_val datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/voc_2012_val_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py voc_2012_trainval datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/voc_2012_trainval_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py voc_2012_test datasets/proposals/MCG-Pascal-Main_trainvaltest_2012-boxes datasets/proposals/voc_2012_test_mcg_proposals_d2.pkl

# COCO 2014 & COCO 2017
python projects/WSOD/tools/convert_proposals.py coco_2014_train datasets/proposals/MCG-COCO-train2014-boxes datasets/proposals/coco_2014_train_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py coco_2014_val datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/coco_2014_val_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py coco_2014_valminusminival datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/coco_2014_valminusminival_mcg_proposals_d2.pkl
python projects/WSOD/tools/convert_proposals.py coco_2014_minival datasets/proposals/MCG-COCO-val2014-boxes datasets/proposals/coco_2014_minival_mcg_proposals_d2.pkl

cd ..
```

## Backbone Models Preparation

You need to initialize from backbone models pre-trained on ImageNet classification tasks, which can be downloaded from [here](https://1drv.ms/f/s!Av60Zz4lbzgWm3AEbq8MW9wEqFUx).

Then you need to move these files into the `detectron2/models/` folder built just before, so that the whole directory structure looks like:

```
detectron2
|_ models
|  |_ WSOD
|  |  |_ ImageNetPretrained
|  |     |_ torchvision
|  |     |  |_ R-18.pkl
|  |     |  |_ R-34.pkl
|  |     |_ ws
|  |        |_ V_ws-16.pkl
|  |        |_ R_ws-18.pkl
|  |        |_ R_ws-50.pkl
|  |        |_ R_ws-101.pkl
|  |_ ...
|_ ...
```

## Getting Started

First, you should get into the `detectron2/` folder.

```
cd detectron2
```

For training, you can follow these commands listed below (VOC07 dataset as example):

### WSDDN

#### VGG16-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/wsddn_V_ws_16_DC5_1x.yaml OUTPUT_DIR output/wsddn_V_ws_16_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet18-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/wsddn_R_ws_18_DC5_1x.yaml OUTPUT_DIR output/wsddn_R_ws_18_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/wsddn_R_ws_50_DC5_1x.yaml OUTPUT_DIR output/wsddn_R_ws_50_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/wsddn_R_ws_101_DC5_1x.yaml OUTPUT_DIR output/wsddn_R_ws_101_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### OICR

#### VGG16-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/oicr_V_ws_16_DC5_1x.yaml OUTPUT_DIR output/oicr_V_ws_16_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet18-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/oicr_R_ws_18_DC5_1x.yaml OUTPUT_DIR output/oicr_R_ws_18_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/oicr_R_ws_50_DC5_1x.yaml OUTPUT_DIR output/oicr_R_ws_50_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/oicr_R_ws_101_DC5_1x.yaml OUTPUT_DIR output/oicr_R_ws_101_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### OICR + bbox reg

#### VGG16-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/oicr_V_ws_16_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_V_ws_16_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet18-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/oicr_R_ws_18_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_R_ws_18_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/oicr_R_ws_50_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_R_ws_50_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/oicr_R_ws_101_DC5_1x.yaml OUTPUT_DIR output/oicr_reg_R_ws_101_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### PCL

#### VGG16-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/pcl_V_ws_16_DC5_1x.yaml OUTPUT_DIR output/pcl_V_ws_16_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet18-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/pcl_R_ws_18_DC5_1x.yaml OUTPUT_DIR output/pcl_R_ws_18_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/pcl_R_ws_50_DC5_1x.yaml OUTPUT_DIR output/pcl_R_ws_50_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/pcl_R_ws_101_DC5_1x.yaml OUTPUT_DIR output/pcl_R_ws_101_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

### PCL + bbox reg

#### VGG16-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/pcl_V_ws_16_DC5_1x.yaml OUTPUT_DIR output/pcl_reg_V_ws_16_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet18-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/pcl_R_ws_18_DC5_1x.yaml OUTPUT_DIR output/pcl_reg_R_ws_18_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/pcl_R_ws_50_DC5_1x.yaml OUTPUT_DIR output/pcl_reg_R_ws_50_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet101-WS

```
python projects/WSOD/tools/train_net.py --num-gpus 4 --config-file projects/WSOD/configs/PascalVOC-Detection/reg/pcl_R_ws_101_DC5_1x.yaml OUTPUT_DIR output/pcl_reg_R_ws_101_DC5_1x_VOC07_`date +'%Y-%m-%d_%H-%M-%S'`
```

## License

WSOD is released under the [Apache 2.0 license](LICENSE).
