import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize


import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import numpy as np
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.common import transform_mesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
import numpy as np
import trimesh,glob,cv2
from im2mesh.layers import ResnetBlockFC
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class generator(nn.Module):
	def __init__(self,  gf_dim):
		super(generator, self).__init__()
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(512, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_8 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_9 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_10 = nn.Linear(self.gf_dim*4, self.gf_dim*4, bias=True)
		self.linear_11 = nn.Linear(self.gf_dim*4, self.gf_dim*4, bias=True)
		self.linear_12 = nn.Linear(self.gf_dim*4, 256,  bias=True)   
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

		nn.init.normal_(self.linear_7.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)
		nn.init.normal_(self.linear_8.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_8.bias,0)
		nn.init.normal_(self.linear_9.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_9.bias,0)
		nn.init.normal_(self.linear_10.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_10.bias,0)
		nn.init.normal_(self.linear_11.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_11.bias,0)
		nn.init.normal_(self.linear_12.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_12.bias,0)
   
   
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
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)
		l7 = F.leaky_relu(l7, negative_slope=0.02, inplace=True)

		l8 = self.linear_8(l7)
		l8 = F.leaky_relu(l8, negative_slope=0.02, inplace=True)

		l9 = self.linear_9(l8)
		l9 = F.leaky_relu(l9, negative_slope=0.02, inplace=True)

		l10 = self.linear_10(l9)
		l10 = F.leaky_relu(l10, negative_slope=0.02, inplace=True)

		l11 = self.linear_11(l10)
		l11 = F.leaky_relu(l11, negative_slope=0.02, inplace=True)

		l12 = self.linear_12(l11)

		return l12
class Decoder(nn.Module):
    ''' Decoder class.

    As discussed in the paper, we implement the OccupancyNetwork
    f and TextureField t in a single network. It consists of 5
    fully-connected ResNet blocks with ReLU activation.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
        out_dim (int): output dimension (e.g. 1 for only
            occupancy prediction or 4 for occupancy and
            RGB prediction)
    '''

    def __init__(self, dim=3, c_dim=256,
                 hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
        super().__init__()
        self.c_dim = 256 #c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules

        #self.fc_pre = generator(64) #nn.Linear(512, 256)
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c=None, batchwise=True, only_occupancy=False,
                only_texture=False, **kwargs):

        assert((len(p.shape) == 3) or (len(p.shape) == 2))

        #p[:]=0.5
        net = self.fc_p(p)
        #print (net, 'p net', self.fc_p.weight)
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c)
                if batchwise:
                    net_c = net_c.unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)
            #print (n, torch.unique(net))

        out = self.fc_out(self.actvn(net))
        #print (out, 'out')

        if only_occupancy:
            if len(p.shape) == 3:
                out = out[:, :, 0]
            elif len(p.shape) == 2:
                out = out[:, 0]
        elif only_texture:
            if len(p.shape) == 3:
                out = out[:, :, 1:4]
            elif len(p.shape) == 2:
                out = out[:, 1:4]

        out = out.squeeze(-1)
        #print ('decoder', torch.unique(out))
        return out

class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, log2_hashmap_size=16, desired_resolution=2048 * self.bound)

        #self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        

        self.sigma_net = Decoder() #MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        self.generator=generator(64)
        #self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else F.softplus



    
        '''parser = argparse.ArgumentParser(
            description='Extract meshes from occupancy process.'
        )
        #parser.add_argument('config', type=str, help='Path to config file.')
    
        
        parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
        parser.add_argument('--upsampling-steps', type=int, default=-1,
                            help='Overrites the default upsampling steps in config')
        parser.add_argument('--refinement-step', type=int, default=-1,
                            help='Overrites the default refinement steps in config')
        parser.add_argument('--text', type=str, default='a blue sofa',
                            help='Text')'''
    
        #args = parser.parse_args()
        cfg = config.load_config('configs/default.yaml')
        is_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if is_cuda else "cpu")
    
        #text = args.text
    
        #text = '/mnt/sda/lzz/out1/'+args.text
        # Overwrite upsamping and refinement step if desired
        '''if args.upsampling_steps != -1:
            cfg['generation']['upsampling_steps'] = args.upsampling_steps
        if args.refinement_step != -1:
            cfg['generation']['refinement_step'] = args.refinement_step'''
    
        # Shortcuts
        '''out_dir = cfg['training']['out_dir']
        generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
        out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
        out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')
    
        batch_size = cfg['generation']['batch_size']
        input_type = cfg['data']['input_type']
        vis_n_outputs = cfg['generation']['vis_n_outputs']
        mesh_extension = cfg['generation']['mesh_extension']
        
        cfg['data']['split_model_for_images'] = False
    
        
        cfg['test']['model_file']='model20.pt'
        dataset = config.get_dataset(cfg, mode='test', return_idx=True)
    
        # Model
        self.model = config.get_model(cfg, device=device, len_dataset=len(dataset))


        checkpoint_io = CheckpointIO(cfg['test']['model_file'], model=self.model)
        checkpoint_io.load(cfg['test']['model_file'], device=device)'''
        
        

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            # self.encoder_bg, self.in_dim_bg = get_encoder('tiledgrid', input_dim=2, num_levels=4, desired_resolution=2048)
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=4)

            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        

        #h = self.encoder(x, bound=self.bound)
        #h = self.sigma_net(h)
        #print ('11111', h.shape, torch.unique(h[:,0]), torch.unique(h[:,1:]))


        c=torch.from_numpy(np.load('c.npy')).cuda()
        c=self.generator(c.float())
        #print ('x', x, x.shape)
        #print (x.shape, c.shape, 'xc.shape')
        h = self.sigma_net(x,c)[0]
        #print ('h', h.shape, torch.unique(h))


        
        #c=torch.from_numpy(np.load('c.npy')).cuda()
        #c=self.model.generator(c.float())
        #print ('c', c)
        
        #print (x.shape, c.shape, 'xc.shape')

        #h = self.model.decoder(x.unsqueeze(0), c)[0] #[:,0,:]  #313344, 4
        
        #h = F.relu(h, inplace=True)
        
        #print ('22222', h.shape, torch.unique(h[:,0]), torch.unique(h[:,1:]))

        sigma = trunc_exp(h[..., 0]  + self.gaussian(x))
        
        #print ('sigma',h[..., 0].shape, sigma.shape )
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal


    def normal(self, x):

        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal

    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        #print ('shading', shading)
        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            #print ('sigma',  torch.unique(sigma))
            normal = None
        
        else:
            # query normal

            sigma, albedo = self.common_forward(x)
            normal = self.normal(x)

            # lambertian shading
            

            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]
            
            #print ('training', sigma.shape, albedo.shape, normal.shape, lambertian.shape)

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params