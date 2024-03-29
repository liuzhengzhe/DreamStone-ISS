# Working with GET3D

## Installation

Please follow [GET3D](https://github.com/nv-tlabs/GET3D) for installation. Download [GET3D models](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW) to "../model" folder. 



## Stage 1

(1) Pretrained Model

We provide pretrained models of stage 1 [here](https://drive.google.com/drive/folders/1BCkpkjVxyGN4XwMDGoWxs9VwE19kZ57s?usp=sharing). 

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

All the examples in paper can be generated immediately at the beginning of the training process. 

For some other cases, taking 'A green rolling chair' as an example, the desired shape can be generated after around 100 iterations. Check the rendered images in "result" folder, and stop training when the desired shape appears. The trained mapper of this example is released [here](https://drive.google.com/drive/folders/1OhGtFmQqE6-R1SwxoXIGKtCMynDpSwHP).

(2) Inference

```
sh test.sh
```


