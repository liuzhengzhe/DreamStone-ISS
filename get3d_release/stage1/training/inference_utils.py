# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
'''
Utily functions for the inference
'''
import torch
import numpy as np
import os
import PIL.Image
from training.utils.utils_3d import save_obj, savemeshtes2
import imageio
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
clip_transform = transforms.Compose([
            transforms.Resize(size=224, max_size=None, antialias=None),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    '''whe=np.where(np.sum(img,2)==0)
    img[:,:,0][whe]=255
    img[:,:,1][whe]=255
    img[:,:,2][whe]=255'''

    gw, gh = grid_size
    _N, C, H, W = img.shape
    gw = _N // gh
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if not fname is None:
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
    return img


def save_3d_shape(mesh_v_list, mesh_f_list, root, idx):
    n_mesh = len(mesh_f_list)
    mesh_dir = os.path.join(root, 'mesh_pred')
    os.makedirs(mesh_dir, exist_ok=True)
    for i_mesh in range(n_mesh):
        mesh_v = mesh_v_list[i_mesh]
        mesh_f = mesh_f_list[i_mesh]
        mesh_name = os.path.join(mesh_dir, '%07d_%02d.obj' % (idx, i_mesh))
        save_obj(mesh_v, mesh_f, mesh_name)


def gen_swap(geo, tex, i,geo_codes, tex_codes,ws_geo_list, ws_tex_list, camera, generator, save_path, gen_mesh=False, ):
    '''
    With two list of latent code, generate a matrix of results, N_geo x N_tex
    :param ws_geo_list: the list of geometry latent code
    :param ws_tex_list: the list of texture latent code
    :param camera:  camera to render the generated mesh
    :param generator: GET3D_Generator
    :param save_path: path to save results
    :param gen_mesh: whether we generate textured mesh
    :return:
    '''
    img_list = []
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        #print (len(ws_geo_list), len(ws_tex_list))
        idx=0
        for i_geo, ws_geo in enumerate(ws_geo_list):
            for i_tex, ws_tex in enumerate(ws_tex_list):
                #print ('wsgeo',ws_geo,ws_tex,'tex',ws_geo.shape,ws_tex.shape)
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                sdf_reg_loss, render_return_value = generator.synthesis.generate(
                    ws_tex.unsqueeze(dim=0), update_emas=None, camera=camera,
                    update_geo=None, ws_geo=ws_geo.unsqueeze(dim=0),
                )
                #print ('unique', torch.unique(img))
                img_list.append(img[:, :3].data.cpu().numpy())
                if gen_mesh:
                    generated_mesh = generator.synthesis.extract_3d_shape(ws_tex.unsqueeze(dim=0), ws_geo.unsqueeze(dim=0))
                    for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                        savemeshtes2(
                            mesh_v.data.cpu().numpy(),
                            all_uvs.data.cpu().numpy(),
                            mesh_f.data.cpu().numpy(),
                            all_mesh_tex_idx.data.cpu().numpy(),
                            os.path.join(save_path, '%02d_%02d.obj' % (i_geo, i_tex))
                        )
                        lo, hi = (-1, 1)
                        img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                        img = (img - lo) * (255 / (hi - lo))
                        img = img.clip(0, 255)
                        mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
                        mask = (mask <= 3.0).astype(np.float)
                        kernel = np.ones((3, 3), 'uint8')
                        dilate_img = cv2.dilate(img, kernel, iterations=1)
                        img = img * (1 - mask) + dilate_img * mask
                        img = img.clip(0, 255).astype(np.uint8)
                        PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                            os.path.join(save_path, '%02d_%02d.png' % (i_geo, i_tex)))

    img_list = np.concatenate(img_list, axis=0)
    img = save_image_grid(img_list, os.path.join(save_path, 'inter_img.jpg'), drange=[-1, 1], grid_size=[ws_tex_list.shape[0], ws_geo_list.shape[0]])
    #print ('save_path',save_path)
    #print (feature.shape)
    #print (geo[i].shape, tex[i].shape)
    np.save(os.path.join(save_path,  'geo.npy'), geo[i].detach().cpu().numpy())
    np.save(os.path.join(save_path,  'tex.npy'), tex[i].detach().cpu().numpy())
    feat=torch.cat((geo_codes[i], tex_codes[i]),-1)
    np.save(os.path.join(save_path,  'feat.npy'), feat.detach().cpu().numpy())



    return img




import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class generator_mapper(nn.Module):
	def __init__(self,  gf_dim, gf_dim2):
		super(generator_mapper, self).__init__()
		self.gf_dim = gf_dim
		self.gf_dim2=gf_dim2
		self.linear_1 = nn.Linear(512, self.gf_dim*128, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*128, self.gf_dim*128, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*128, self.gf_dim*128, bias=True)

		self.linear_1g1 = nn.Linear(512, self.gf_dim*128*4, bias=True)
		self.linear_2g1 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_3g1 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_4g1 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_5g1 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_6g1 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_7g1 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128, bias=True)

		'''self.linear_1g2 = nn.Linear(512, self.gf_dim*128*4, bias=True)
		self.linear_2g2 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_3g2 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_4g2 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_5g2 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_6g2 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128*4, bias=True)
		self.linear_7g2 = nn.Linear(self.gf_dim*128*4, self.gf_dim*128, bias=True)'''


		self.linear_4 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_8 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_9 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_10 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_11 = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_12 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_13 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_14 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_15 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_16 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_17 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_18 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_19 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_20 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_21 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_22 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_23 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_24 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		self.linear_25 = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 
		#self.linear_26 = nn.Linear(self.gf_dim*128, self.gf_dim*128,  bias=True) 


		self.linear_4x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_5x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_6x = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True)
		self.linear_7x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_8x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_9x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_10x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_11x = nn.Linear(self.gf_dim*128, self.gf_dim2*8, bias=True)
		self.linear_12x = nn.Linear(self.gf_dim*128, self.gf_dim2*8,  bias=True) 



		self.norm1 = nn.LayerNorm(self.gf_dim*128,elementwise_affine=False) 
		self.norm2 = nn.LayerNorm(self.gf_dim*128,elementwise_affine=False) 
		self.norm3 = nn.LayerNorm(self.gf_dim*128,elementwise_affine=False) 

		self.norm1g1 = nn.LayerNorm(self.gf_dim*128*4,elementwise_affine=False) 
		self.norm2g1 = nn.LayerNorm(self.gf_dim*128*4,elementwise_affine=False) 
		self.norm3g1 = nn.LayerNorm(self.gf_dim*128*4,elementwise_affine=False) 
		self.norm4g1 = nn.LayerNorm(self.gf_dim*128*4,elementwise_affine=False) 
		self.norm5g1 = nn.LayerNorm(self.gf_dim*128*4,elementwise_affine=False) 
		self.norm6g1 = nn.LayerNorm(self.gf_dim*128*4,elementwise_affine=False) 
		self.norm7g1 = nn.LayerNorm(self.gf_dim*128,elementwise_affine=False) 


		'''s=0.01
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=s)
		nn.init.constant_(self.linear_6.bias,0)

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
		nn.init.constant_(self.linear_12.bias,0)'''

   
	def forward(self, clip_feature):

		#print (clip_feature.shape,'clip')

		l1g1 = self.norm1g1(self.linear_1g1(clip_feature))
		l1g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)

		l1g1 = self.norm2g1(self.linear_2g1(l1g1))
		l1g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)

		l1g1 = self.norm3g1(self.linear_3g1(l1g1))
		l1g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)

		l1g1 = self.norm4g1(self.linear_4g1(l1g1))
		l1g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)

		l1g1 = self.norm5g1(self.linear_5g1(l1g1))
		l1g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)

		l1g1 = self.norm6g1(self.linear_6g1(l1g1))
		l1g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)

		l1g1 = self.norm7g1(self.linear_7g1(l1g1))
		l2g1 = F.leaky_relu(l1g1, negative_slope=0.02, inplace=True)





		l4 = self.linear_4(l2g1)
		#l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l2g1)
		#l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l2g1)
		#l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l2g1)
		#l7 = F.leaky_relu(l7, negative_slope=0.02, inplace=True)

		l8 = self.linear_8(l2g1)
		#l8 = F.leaky_relu(l8, negative_slope=0.02, inplace=True)

		l9 = self.linear_9(l2g1)
		#l9 = F.leaky_relu(l9, negative_slope=0.02, inplace=True)

		l10 = self.linear_10(l2g1)
		#l10 = F.leaky_relu(l10, negative_slope=0.02, inplace=True)

		l11 = self.linear_11(l2g1)
		#l11 = F.leaky_relu(l11, negative_slope=0.02, inplace=True)

		l12 = self.linear_12(l2g1)


		l13 = self.linear_13(l2g1)
		#l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l14 = self.linear_14(l2g1)
		#l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l15 = self.linear_15(l2g1)
		#l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l16 = self.linear_16(l2g1)
		#l7 = F.leaky_relu(l7, negative_slope=0.02, inplace=True)

		l17 = self.linear_17(l2g1)
		#l8 = F.leaky_relu(l8, negative_slope=0.02, inplace=True)

		l18 = self.linear_18(l2g1)
		#l9 = F.leaky_relu(l9, negative_slope=0.02, inplace=True)

		l19 = self.linear_19(l2g1)
		#l10 = F.leaky_relu(l10, negative_slope=0.02, inplace=True)

		l20 = self.linear_20(l2g1)
		#l11 = F.leaky_relu(l11, negative_slope=0.02, inplace=True)

		l21 = self.linear_21(l2g1)

		l22 = self.linear_22(l2g1)
		l23 = self.linear_23(l2g1)
		l24 = self.linear_24(l2g1)
		l25 = self.linear_25(l2g1)
		#l26 = self.linear_26(l2)



		g1=torch.cat((l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14),-1)
		g2=torch.cat((l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25),-1)


		l1 = self.norm1(self.linear_1(clip_feature))
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.norm2(self.linear_2(l1))
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.norm3(self.linear_3(l2))
		l2 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)



		l4 = self.linear_4x(l2)
		#l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5x(l2)
		#l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6x(l2)
		#l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7x(l2)
		#l7 = F.leaky_relu(l7, negative_slope=0.02, inplace=True)

		l8 = self.linear_8x(l2)
		#l8 = F.leaky_relu(l8, negative_slope=0.02, inplace=True)

		l9 = self.linear_9x(l2)
		#l9 = F.leaky_relu(l9, negative_slope=0.02, inplace=True)

		l10 = self.linear_10x(l2)
		#l10 = F.leaky_relu(l10, negative_slope=0.02, inplace=True)

		l11 = self.linear_11x(l2)
		#l11 = F.leaky_relu(l11, negative_slope=0.02, inplace=True)

		l12 = self.linear_12x(l2)
		#g = self.linear_g(l3)
		#c = self.linear_c(l3)
		c=torch.cat((l4,l5,l6,l7,l8,l9,l10,l11,l12),-1)
		return g1, g2, c



def save_visualization_for_interpolation(
        generator, num_sam=4000, c_to_compute_w_avg=None, save_dir=None, gen_mesh=False):
    '''
    Interpolate between two latent code and generate a swap between them
    :param generator: GET3D generator
    :param num_sam: number of samples we hope to generate
    :param c_to_compute_w_avg: None is default
    :param save_dir: path to save
    :param gen_mesh: whether we want to generate 3D textured mesh
    :return:
    '''
    with torch.no_grad():
        print ('3d util')
        generator.update_w_avg(c_to_compute_w_avg)
        geo_codes = torch.randn(num_sam, generator.z_dim, device=generator.device)
        tex_codes = torch.randn(num_sam, generator.z_dim, device=generator.device)
        #print ('code', geo_codes.shape, tex_codes.shape)

        '''model = generator(64).cuda()
        model.load_state_dict(torch.load('models/model382.pt'), strict=False)
        model.eval()

        import clip
        from PIL import Image
        clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

        image_data = Image.open(path).convert('RGB')
        clip_image = clip_transform(image_data)
        clip_feature=clip_model.encode_image(clip_image)
        pred_feature=model(clip_feature.detach())
        geo_codes=pred_feature[0:1,:512]
        tex_codes=pred_feature[0:1,512:]'''


        #print (geo_codes.shape)
        ws_geo = generator.mapping_geo(geo_codes, None, truncation_psi=0.7)
        #print (ws_geo.shape)
        ws_tex = generator.mapping(tex_codes, None, truncation_psi=0.7)
        


        '''text='a red tall chair'
        device='cuda'
        import clip
        text = clip.tokenize(text).to(device)

        model, preprocess = clip.load("ViT-B/32", device=device)
        text_features = model.encode_text(text)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #import train_layernorm_multi_g2
        mapper = generator_mapper(4,64).cuda()
        mapper.load_state_dict(torch.load('model_batch_1_2894.pt'), strict=True)
        mapper.eval()

        g1, g2, c=mapper(text_features.detach().float())
        g1=torch.reshape(g1, (1,11,512))
        g2=torch.reshape(g2, (1,11,512))
        c=torch.reshape(c, (1,9,512))


        ws_geo=torch.cat((g1,g2),1)
        ws_tex=c'''

        camera_list = [generator.synthesis.generate_rotate_camera_list(n_batch=num_sam)[4]]

        select_geo_codes = np.arange(2000)  # You can change to other selected shapes
        select_tex_codes = np.arange(2000)
        for i in range(len(select_geo_codes) ):
            ws_geo_a = ws_geo[select_geo_codes[i]].unsqueeze(dim=0)
            #ws_geo_b = ws_geo[select_geo_codes[i + 1]].unsqueeze(dim=0)
            ws_tex_a = ws_tex[select_tex_codes[i]].unsqueeze(dim=0)
            #ws_tex_b = ws_tex[select_tex_codes[i + 1]].unsqueeze(dim=0)
            #print ('ws',ws_geo_a.shape, ws_geo_b.shape, ws_tex_a.shape, ws_tex_b.shape)
            new_ws_geo = []
            new_ws_tex = []
            n_interpolate = 1
            for _i in range(n_interpolate):
                w = float(_i + 1) / n_interpolate
                #w = 1 - w
                new_ws_geo.append(ws_geo_a) # * w + ws_geo_b * (1 - w))
                new_ws_tex.append(ws_tex_a) # * w + ws_tex_b * (1 - w))
            new_ws_tex = torch.cat(new_ws_tex, dim=0)
            new_ws_geo = torch.cat(new_ws_geo, dim=0)
            save_path = os.path.join(save_dir, 'generate_%02d' % (i))
            os.makedirs(save_path, exist_ok=True)
            gen_swap(ws_geo, ws_tex, i, geo_codes, tex_codes,
                new_ws_geo, new_ws_tex, camera_list[0], generator,
                save_path=save_path, gen_mesh=gen_mesh
            )


def save_visualization(
        G_ema, grid_z, grid_c, run_dir, cur_nimg, grid_size, cur_tick,
        image_snapshot_ticks=50,
        save_gif_name=None,
        save_all=True,
        grid_tex_z=None,
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg()
        camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
        camera_img_list = []
        if not save_all:
            camera_list = [camera_list[4]]  # we only save one camera for this
        if grid_tex_z is None:
            grid_tex_z = grid_z
        for i_camera, camera in enumerate(camera_list):
            images_list = []
            mesh_v_list = []
            mesh_f_list = []
            for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                    z=z, geo_z=geo_z, c=c, noise_mode='const',
                    generate_no_light=True, truncation_psi=0.7, camera=camera)
                rgb_img = img[:, :3]
                save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1)], dim=-1).detach()
                images_list.append(save_img.cpu().numpy())
                mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])
            images = np.concatenate(images_list, axis=0)
            if save_gif_name is None:
                save_file_name = 'fakes'
            else:
                save_file_name = 'fakes_%s' % (save_gif_name.split('.')[0])
            if save_all:
                img = save_image_grid(
                    images, None,
                    drange=[-1, 1], grid_size=grid_size)
            else:
                img = save_image_grid(
                    images, os.path.join(
                        run_dir,
                        f'{save_file_name}_{cur_nimg // 1000:06d}_{i_camera:02d}.png'),
                    drange=[-1, 1], grid_size=grid_size)
            camera_img_list.append(img)
        if save_gif_name is None:
            save_gif_name = f'fakes_{cur_nimg // 1000:06d}.gif'
        if save_all:
            imageio.mimsave(os.path.join(run_dir, save_gif_name), camera_img_list)
        n_shape = 10  # we only save 10 shapes to check performance
        if cur_tick % min((image_snapshot_ticks * 20), 100) == 0:
            save_3d_shape(mesh_v_list[:n_shape], mesh_f_list[:n_shape], run_dir, cur_nimg // 100)


def save_textured_mesh_for_inference(
        G_ema, grid_z, grid_c, run_dir, save_mesh_dir=None,
        c_to_compute_w_avg=None, grid_tex_z=None, use_style_mixing=False):
    '''
    Generate texture mesh for generation
    :param G_ema: GET3D generator
    :param grid_z: a grid of latent code for geometry
    :param grid_c: None
    :param run_dir: save path
    :param save_mesh_dir: path to save generated mesh
    :param c_to_compute_w_avg: None
    :param grid_tex_z: latent code for texture
    :param use_style_mixing: whether we use style mixing or not
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg(c_to_compute_w_avg)
        save_mesh_idx = 0
        mesh_dir = os.path.join(run_dir, save_mesh_dir)
        os.makedirs(mesh_dir, exist_ok=True)
        for idx in range(len(grid_z)):
            geo_z = grid_z[idx]
            if grid_tex_z is None:
                tex_z = grid_z[idx]
            else:
                tex_z = grid_tex_z[idx]
            generated_mesh = G_ema.generate_3d_mesh(
                geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7,
                use_style_mixing=use_style_mixing)
            for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                savemeshtes2(
                    mesh_v.data.cpu().numpy(),
                    all_uvs.data.cpu().numpy(),
                    mesh_f.data.cpu().numpy(),
                    all_mesh_tex_idx.data.cpu().numpy(),
                    os.path.join(mesh_dir, '%07d.obj' % (save_mesh_idx))
                )
                lo, hi = (-1, 1)
                img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                img = (img - lo) * (255 / (hi - lo))
                img = img.clip(0, 255)
                mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
                mask = (mask <= 3.0).astype(np.float)
                kernel = np.ones((3, 3), 'uint8')
                dilate_img = cv2.dilate(img, kernel, iterations=1)
                img = img * (1 - mask) + dilate_img * mask
                img = img.clip(0, 255).astype(np.uint8)
                PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                    os.path.join(mesh_dir, '%07d.png' % (save_mesh_idx)))
                save_mesh_idx += 1


def save_geo_for_inference(G_ema, run_dir):
    '''
    Generate the 3D objs (without texture) for generation
    :param G_ema: GET3D Generation
    :param run_dir: save path
    :return:
    '''
    import kaolin as kal
    def normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample, normalized_scale=1.0):
        vertices = mesh_v.cuda()
        scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
        mesh_v1 = vertices / scale * normalized_scale
        mesh_f1 = mesh_f.cuda()
        points, _ = kal.ops.mesh.sample_points(mesh_v1.unsqueeze(dim=0), mesh_f1, n_sample)
        return points

    with torch.no_grad():
        use_style_mixing = True
        truncation_phi = 1.0
        mesh_dir = os.path.join(run_dir, 'gen_geo_for_eval_phi_%.2f' % (truncation_phi))
        surface_point_dir = os.path.join(run_dir, 'gen_geo_surface_points_for_eval_phi_%.2f' % (truncation_phi))
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(surface_point_dir, exist_ok=True)
        n_gen = 1500 * 5  # We generate 5x of test set here
        i_mesh = 0
        for i_gen in tqdm(range(n_gen)):
            geo_z = torch.randn(1, G_ema.z_dim, device=G_ema.device)
            generated_mesh = G_ema.generate_3d_mesh(
                geo_z=geo_z, tex_z=None, c=None, truncation_psi=truncation_phi,
                with_texture=False, use_style_mixing=use_style_mixing)
            for mesh_v, mesh_f in zip(*generated_mesh):
                if mesh_v.shape[0] == 0: continue
                save_obj(mesh_v.data.cpu().numpy(), mesh_f.data.cpu().numpy(), os.path.join(mesh_dir, '%07d.obj' % (i_mesh)))
                points = normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample=2048, normalized_scale=1.0)
                np.savez(os.path.join(surface_point_dir, '%07d.npz' % (i_mesh)), pcd=points.data.cpu().numpy())
                i_mesh += 1
