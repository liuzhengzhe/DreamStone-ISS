import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
from im2mesh.checkpoints import CheckpointIO
from PIL import Image

clip_transform = transforms.Compose([
            transforms.Resize(size=224, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

dvr_transform = transforms.Compose([
    #transforms.Resize(img_size),
    transforms.ToTensor()])

def default_loader(path):
    #im=torch.from_numpy(cv2.imread(path)).cuda()
    #img_tensor=torch.tensor(ims.astype('float32')).permute(2,0,1)
    image_data = Image.open(path).convert('RGB')
    clip_image = clip_transform(image_data)
    #dvr_image=dvr_transform(image_data)
    return clip_image#, clip_image


class customData(Dataset):
    def __init__(self, dataset = '', data_transforms=None, loader = default_loader):
        self.img_name=glob.glob('/mnt/sdc/lzz/ShapeNet/*/*/image/*.png')
        #self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        clip_image = self.loader(img_name)
        #print (torch.unique(clip_image), torch.unique(dvr_image))
        #if self.data_transforms is not None:
        #clip_image, dvr_image= self.data_transforms[self.dataset](img)
          

        return clip_image#, dvr_image
 
image_datasets = customData() 
dataloders =  torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=512,
                                                 shuffle=True) 
                                                 
                                                 
                                                 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F




class generator(nn.Module):
	def __init__(self,  gf_dim):
		super(generator, self).__init__()
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(512, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*4, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*4, 256, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)


   
   
	def forward(self, clip_feature, is_training=False):

		l1 = self.linear_1(clip_feature)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)

		return l6

#model = generator(64).cuda()
##model.load_state_dict(torch.load('state_dict.pt'))
#model.train()
import resnet18 as resnet
model = resnet.resnet18(pretrained=True) #resnet18_gram.resnet18() #pretrained=True)
model.load_state_dict(resnetinit.state_dict(),strict=False)
model.train()

import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

import clip
clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

from im2mesh import config
cfg = config.load_config('configs/demo/demo_combined.yaml', 'configs/default.yaml')
dvr_model=config.get_model(cfg, device='cuda')
dvr_model.load_state_dict(torch.load("ours_combined-af2bce07.pt")['model'])


#checkpoint_io = CheckpointIO('.', model=model, optimizer=optimizer)
#load_dict = checkpoint_io.load('ours_combined-af2bce07.pt', device='cuda')

for param in dvr_model.parameters():
  param.requires_grad=False

for param in clip_model.parameters():
  param.requires_grad=False

dvr_model.eval()
clip_model.eval()

clip_criterion=torch.nn.CosineSimilarity()

device='cuda'
epochs=99999999

i=0
running_loss=0.0
for epoch in range(3,epochs):
    for clip_image, dvr_image in dataloders:
        i+=1
        clip_image, dvr_image = clip_image.to(device), dvr_image.to(device)
        optimizer.zero_grad()
        clip_feature=clip_model.encode_image(clip_image)


        random_noise = torch.randn(clip_feature.shape).to(device)  # random Gaussian
        random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True) # Normalize to sphere
        clip_feature = clip_feature*(1-0.75) + random_noise*0.75
        clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
        pred_feature=model(clip_feature.detach())
        
        gt=dvr_model.encoder(dvr_image)
        
        loss = torch.sum(torch.abs(pred_feature-gt)) #torch.sum(1-clip_criterion(pred_feature, gt.detach()))*40 
        #print (loss,'loss',epoch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1== 0:   # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/1 :.3f}')
            running_loss = 0.0
            torch.save(model.state_dict(), 'model'+str(epoch)+'.pt')