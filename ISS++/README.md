# ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation

Code for the paper [ISS++: Image as Stepping Stone for Text-Guided 3D Shape Generation]().

**Authors**: Zhengzhe Liu, Peng Dai, Ruihui Li, Xiaojuan Qi, Chi-Wing Fu

## Installation

Follow [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion)


##  Train

(1) First generate a coarse shape for initialization, e.g., "a police car" or "a red car" in "stage2". 

(2) Training

SDS-Guided Refinement: refine the output from two-stage feature-space alignment. 
```
python main.py --source "a police car" --text "a police car" --workspace 'a police car' -O
```

Out-of Vocabulary Categories: generate out-of-vocabulary 3D shapes, like "hamburger". 
```
python main.py --source "a red car" --text "a hamburger" --workspace 'a hamburger' -O
```

--source: the initialized shape
--text: the text description of your desired 3D shape
--workspace: the folder name

Check the "validation" folder. In most cases, the desired shape can be generated in around 30 to 50 epochs. 


(3) Inference

```
python main.py --source "a red car" --text "a hamberger" --workspace 'a hamberger' -O --test --save_mesh
```
