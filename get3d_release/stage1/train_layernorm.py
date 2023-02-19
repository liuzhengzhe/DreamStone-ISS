from xml.sax.handler import feature_external_ges
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob, os

from PIL import Image

clip_transform = transforms.Compose([
            transforms.Resize(size=224, max_size=None, antialias=None),
            #transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

dvr_transform = transforms.Compose([
    #transforms.Resize(img_size),
    transforms.ToTensor()])

def default_loader(path):
    #im=torch.from_numpy(cv2.imread(path)).cuda()
    #img_tensor=torch.tensor(ims.astype('float32')).permute(2,0,1)
    #image_data = Image.open(path).convert('RGB')
    #clip_image = clip_transform(image_data)
    #dvr_image=dvr_transform(image_data)
    image = preprocess(Image.open(path)).to(device)
    return image, image


class customData(Dataset):
    def __init__(self, dataset = '', data_transforms=None, loader = default_loader):
        self.img_name=glob.glob('save_inference_results/shapenet_chair/inference/interpolation/*')[:20]
        #self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]+'/inter_img.jpg'
        clip_image, dvr_image = self.loader(img_name)
        #print ('im',torch.unique(clip_image))
        g=np.load(self.img_name[item]+'/geo3.npy')
        c=np.load(self.img_name[item]+'/tex3.npy')
        g=np.reshape(g,(1,22*32))
        c=np.reshape(c,(1,9*32))


        #print (torch.unique(clip_image), torch.unique(dvr_image))
        #if self.data_transforms is not None:
        #clip_image, dvr_image= self.data_transforms[self.dataset](img)
          

        return clip_image,g,c
 
image_datasets = customData() 
dataloders =  torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=32,
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
		self.linear_4x = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_5x = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_6x = nn.Linear(self.gf_dim*8, self.gf_dim*8,  bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_8 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_9 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_10 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_11 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_12 = nn.Linear(self.gf_dim*8, self.gf_dim*8,  bias=True) 

		self.norm1 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm2 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm3 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm4 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm5 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm6 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm7 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm8 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm9 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm10 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm11 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 
		self.norm12 = nn.LayerNorm(self.gf_dim*8,elementwise_affine=False) 

		self.linear_g = nn.Linear(self.gf_dim*8, 32*22,  bias=True) 
		self.linear_c = nn.Linear(self.gf_dim*8, 32*9,  bias=True) 


		s=0.01
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4x.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_4x.bias,0)
		nn.init.normal_(self.linear_5x.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_5x.bias,0)
		nn.init.normal_(self.linear_6x.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_6x.bias,0)

		nn.init.normal_(self.linear_7.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_7.bias,0)
		nn.init.normal_(self.linear_8.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_8.bias,0)
		nn.init.normal_(self.linear_9.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_9.bias,0)
		nn.init.normal_(self.linear_10.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_10.bias,0)
		nn.init.normal_(self.linear_11.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_11.bias,0)
		nn.init.normal_(self.linear_12.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_12.bias,0)
		nn.init.normal_(self.linear_g.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_g.bias,0)
		nn.init.normal_(self.linear_c.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_c.bias,0)
   
	def forward(self, clip_feature, is_training=False):

		l1 = self.norm1(self.linear_1(clip_feature))
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.norm2(self.linear_2(l1))
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.norm3(self.linear_3(l2))
		'''l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.norm4(self.linear_4x(l3))
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.norm5(self.linear_5x(l4))
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.norm6(self.linear_6x(l5))
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.norm7(self.linear_7(l6))
		l7 = F.leaky_relu(l7, negative_slope=0.02, inplace=True)

		l8 = self.norm8(self.linear_8(l7))
		l8 = F.leaky_relu(l8, negative_slope=0.02, inplace=True)

		l9 = self.norm9(self.linear_9(l8))
		l9 = F.leaky_relu(l9, negative_slope=0.02, inplace=True)

		l10 = self.norm10(self.linear_10(l9))
		l10 = F.leaky_relu(l10, negative_slope=0.02, inplace=True)

		l11 = self.norm11(self.linear_11(l10))
		l11 = F.leaky_relu(l11, negative_slope=0.02, inplace=True)

		l12 = self.norm12(self.linear_12(l11))'''
		g = self.linear_g(l3)
		c = self.linear_c(l3)
		return g,c


model = generator(512).cuda()

#model.load_state_dict(torch.load('/mnt/sdc/lzz/bigmapper_test_2stage2_disc/model.pt'), strict=False)
model.train()
import torch.optim as optim

#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

import clip
clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

#from im2mesh import config
#cfg = config.load_config('configs/demo/demo_combined.yaml', 'configs/default.yaml')
#dvr_model=config.get_model(cfg, device='cuda')
#dvr_model.load_state_dict(torch.load("ours_combined-af2bce07.pt")['model'])


#checkpoint_io = CheckpointIO('.', model=model, optimizer=optimizer)
#load_dict = checkpoint_io.load('ours_combined-af2bce07.pt', device='cuda')

#for param in dvr_model.parameters():
#  param.requires_grad=False

for param in clip_model.parameters():
  param.requires_grad=False

#dvr_model.eval()
clip_model.eval()

clip_criterion=torch.nn.CosineSimilarity()

device='cuda'
epochs=99999999

i=0
running_loss=0.0
for epoch in range(3,epochs):
	for clip_image, g_gt,c_gt in dataloders:
		i+=1
		clip_image, g_gt, c_gt = clip_image.to(device), g_gt.to(device), c_gt.to(device)
		optimizer.zero_grad()
		clip_feature=clip_model.encode_image(clip_image).float()

		'''random_noise = torch.randn(clip_feature.shape).to(device)  # random Gaussian
		random_noise = random_noise/random_noise.norm(dim=-1, keepdim=True) # Normalize to sphere
		clip_feature = clip_feature*(1-0.75) + random_noise*0.75'''
		clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
		#print ('clip',clip_feature, g_gt)
		g,c=model(clip_feature.detach()*10)

		#gt=feat #dvr_model.encoder(dvr_image)
		#print (torch.unique(g)) #, torch.unique(g_gt) )
		gloss = torch.mean(torch.abs(g-g_gt)) #torch.sum(1-clip_criterion(pred_feature, gt.detach()))*40 
		closs = torch.mean(torch.abs(c-c_gt))
		print ('loss', gloss, closs, epoch)
		loss=gloss+closs
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		#print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/i :.3f}')
		if epoch % 2 == 0: 
			torch.save(model.state_dict(), 'modelsmall_norm'+str(epoch)+'.pt')
			try:
				os.remove('modelsmall_norm'+str(epoch-4)+'.pt')
			except:
				pass