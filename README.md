## Introduction

This repository contains code to reproduce Cityscapes semantic segmentation result described in the paper: Confidence Adaptive Anytime Pixel-Level Recognition. It is based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1).

## Quick start
### 1. Install
1. Install PyTorch=1.1.0 following the [official instructions](https://pytorch.org/)
2. git clone $ROOT
3. Install dependencies: pip install -r requirements.txt

### 2. Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) dataset and place a symbolic link of the dataset under `data` folder.



Your directory tree should look like this:
````
$ROOT/data
└── cityscapes
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    └── leftImg8bit
        ├── test
        ├── train
        └── val

````

### 3. Pretrained model preparation
1. create a folder called `pretrained_models` under this repository.
2. You need to download the [HRNet-W18-C-Small-v2](https://1drv.ms/u/s!Aus8VCZ_C_33gRmfdPR79WBS61Qn?e=HVZUi8) and [HRNet-W48-C](https://1drv.ms/u/s!Aus8VCZ_C_33dKvqI6pBZlifgJk) from [HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification.git) and place these two pretrained models under `pretrained_models`.


## Train and test

### Train
There are two options for the backbone HRnet model, W18 and W48. They are configured and specified with their respected experiment files `w48.yaml` and `w18.yaml` under `experimens/cityscapes`.
The w18_and_w48_experiments.sh file contains commands to train each of the settings mentioned in the paper. Specifically,

To run the Early Exits (EE) setting: use

````bash
python -m torch.distributed.launch tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output_new/w48/RH/    MODEL.NAME model_anytime  TRAIN.END_EPOCH 484 
````


To run the Redesigned Heads (RH) setting: use 

````bash
python -m torch.distributed.launch tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output/w48/RH/    MODEL.NAME model_anytime   EXIT.TYPE 'downup_pool_1x1_inter_flexible' EXIT.FIX_INTER_CHANNEL True  EXIT.INTER_CHANNEL 64 TRAIN.END_EPOCH 484  
````


To run Confidence Adatative (CA) setting: use

````bash
python -m torch.distributed.launch tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output/w48/CA/0.998    MODEL.NAME model_anytime   TRAIN.END_EPOCH 484 MASK.FULL_USE True   MASK.CONF_THRE 0.998
````


To run EE + RH + CA: use

````bash
python -m torch.distributed.launch tools/train_ee.py --cfg experiments/cityscapes/w48.yaml OUTPUT_DIR output_new/w48/FULL    MODEL.NAME model_anytime  TRAIN.END_EPOCH 484 EXIT.TYPE 'downup_pool_1x1_inter_flexible' EXIT.FIX_INTER_CHANNEL True  EXIT.INTER_CHANNEL 64 MASK.FULL_USE True   MASK.CONF_THRE 0.998
````

After the training finishes, the test result will be stored in the directory specified by the argument `OUTPUT_DIR` in the YAML configuration file. Inside it, you will find:
1) `result.txt`: contains mIOU for each exit and the average mIOU of the four exits. 

2) `test_stats.json`: contains computation related statistics including number of parameters of the full model and flops.

3) `final_state.pth`: the trained model.

4) `config.yaml`: the configuration file used during this training session.



### Test

1. To evaluate the trained model on Cityscapes validation set, the easiest way is to run: 

`python tools/test_ee.py --cfg <Your output directoy>/config.yaml`

This command will use the `final_state.pth` in your output directory.

2. To evaluate arbitary models, specify the configuration file, the location of the model and other desired test settings (scale and flip). For example:


`python tools/test.py --cfg experiments/cityscapes/<Your config file>.yaml \
                     TEST.MODEL_FILE <Your model>.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True`

***

## Command line argument explanation
**MASK.CONF_THRE**: the confidence thershold used in CA described in paper.

**MODEL.EXTRA.EE_WEIGHTS**: weighing parameter for loss obtained at exits that's used in training. ex: '(1,1,1,1)' means  loss obtained from the 1st, 2nd, 3rd and final exits have equal weights.
