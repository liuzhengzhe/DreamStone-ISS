import torch
import clip,cv2
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print (cv2.imread('images.jpg'))
image = preprocess(Image.open("images.jpg")).unsqueeze(0).to(device)
print ('image',torch.unique(image)) #-1.79, 2.14

-0.45 /0.26