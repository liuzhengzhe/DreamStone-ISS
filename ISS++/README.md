# ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation

Code for the paper [ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation]().

**Authors**: Zhengzhe Liu, Peng Dai, Ruihui Li, Xiaojuan Qi, Chi-Wing Fu

## Installation

Follow [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)


##  Train

(1) First generate a coarse shape for initialization, e.g., "a red car", in "stage2". 

(2) Training

```
sh train.sh
```

--source: the initialized shape

Check the "validation" folder. In most cases, the desired shape can be generated in around 30 to 50 epochs. 


(3) Inference

```
sh test.sh
```
