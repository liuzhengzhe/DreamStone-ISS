# Working with GET3D

## Installation

Please follow [GET3D](https://github.com/nv-tlabs/GET3D) for installation, and download [GET3D models](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW) to "../model" folder. 

## Stage 1

(1) Pretrained Model

We provide pretrained models of stage 1 [here](). 

(2) Prepare for rendered images as training data. Take the chair category as an example. 

```
cd stage1
sh test.sh
```

Results are in "save_inference_results/shapenet_chair/inference/interpolation/"

(3) Training

```
python train_chair.py
```

## Stage 2

(1) Training

Modify --text and --stage1_model accordingly. 

```
sh train.sh
```

Take 'A blue swivel chair' as an example, the initial image is like the below left, and we can get the result after around 300 to 400 iterations. 


