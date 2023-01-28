import os
import torch
from im2mesh.common import (
    check_weights, get_tensor_values, transform_to_world,
    transform_to_camera_space, sample_patch_points, arange_pixels,
    make_3d_grid, compute_iou, get_occupancy_loss_points,
    get_freespace_loss_points
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
from tqdm import tqdm
import logging
from im2mesh import losses
logger_py = logging.getLogger(__name__)
import numpy as np
import cv2,glob,clip,random


class Trainer(BaseTrainer):
    ''' Trainer object for the DVR.

    Args:
        model (nn.Module): DVR model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        threshold (float): threshold value
        n_training_points (int): number of training points
        n_eval_points (int): number of evaluation points
        lambda_occupied (float): lambda for occupancy loss
        lambda_freespace (float): lambda for freespace loss
        lambda_rgb (float): lambda for rgb loss
        lambda_normal (float): lambda for normal loss
        lambda_depth (float): lambda for depth loss
        lambda_image_gradient: lambda for image gradient loss
        lambda_sparse_depth (float): lambda for sparse depth loss
        generator (Object): Generator object for visualization
        patch_size (int): training patch size
        reduction_method (str): reduction method for losses (default: sum)
        sample_continuous (bool): whether to sample pixels continuously in
            range [-1, 1] or only at pixel location
        overwrite_visualizations( bool): whether to overwrite files in
            visualization folder. Default is true, modify this if you want to
            save the outputs for a progression over training iterations
        depth_from_visual_hull (bool): whether to use depth from visual hull
            for occupancy loss
        depth_range (float): depth range; if cube intersection is
            used this value is not relevant
        depth_loss_on_world_points (bool): whether the depth loss should be
            applied on the world points (see SupMat for details)
        occupancy_random_normal (bool): whether to sample from a normal
            distribution instead of uniform for occupancy loss
        use_cube_intersection (bool): whether to use ray intersections with
            unit cube for losses
        always_freespace (bool): whether to always apply the freespace loss
        multi_gpu (bool): whether to use multiple GPUs for training
    '''

    def __init__(self, model, optimizer, device=None, vis_dir=None,
                 threshold=0.5, n_training_points=2048, n_eval_points=4000,
                 lambda_occupied=1., lambda_freespace=1., lambda_rgb=1.,
                 lambda_normal=0.05, lambda_depth=0., lambda_image_gradients=0,
                 lambda_sparse_depth=0., generator=None, patch_size=1,
                 reduction_method='sum', sample_continuous=False,
                 overwrite_visualization=True,
                 depth_from_visual_hull=False, depth_range=[0, 2.4],
                 depth_loss_on_world_points=False,
                 occupancy_random_normal=False,
                 use_cube_intersection=False, always_freespace=True,
                 multi_gpu=False, **kwargs):
        self.model = model
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        self.optimizer = optimizer
        self.device = device
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.lambda_occupied = lambda_occupied
        self.lambda_freespace = lambda_freespace
        self.lambda_rgb = lambda_rgb
        self.generator = generator
        self.n_eval_points = n_eval_points
        self.lambda_depth = lambda_depth
        self.lambda_image_gradients = lambda_image_gradients
        self.patch_size = patch_size
        self.reduction_method = reduction_method
        self.sample_continuous = sample_continuous
        self.lambda_sparse_depth = lambda_sparse_depth
        self.overwrite_visualization = overwrite_visualization
        self.depth_from_visual_hull = depth_from_visual_hull
        self.depth_range = depth_range
        self.depth_loss_on_world_points = depth_loss_on_world_points
        self.occupancy_random_normal = occupancy_random_normal

        self.use_cube_intersection = use_cube_intersection
        self.always_freespace = always_freespace
        self.multi_gpu = multi_gpu
        self.lambda_normal = lambda_normal

        self.n_training_points = n_training_points

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        
        #self.clip_model, preprocess = clip.load("ViT-B/32", device=device)
        self.clip_criterion=torch.nn.CosineSimilarity()



    def train_step(self, data,c, condition1, text, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss,clip_loss1, clip_loss2,dis = self.compute_loss(data,c, condition1, text, it=it)
        # loss.backward(retain_graph=True)
        loss.backward()
        check_weights(self.model.state_dict())
        self.optimizer.step()

        return loss.item(),clip_loss1.item(), clip_loss2.item(),dis.item()

    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict = {}

        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True)

        for (k, v) in eval_dict.items():
            eval_dict[k] = v.item()

        return eval_dict

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
        img = data.get('img').to(device)
        mask_img = data.get('img.mask').unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        depth_img = data.get('img.depth', torch.empty(1, 0)
                             ).unsqueeze(1).to(device)
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        # Get sparse point data
        if self.lambda_sparse_depth != 0:
            sparse_depth = {}
            sparse_depth['p'] = data.get('sparse_depth.p_img').to(device)
            sparse_depth['p_world'] = data.get(
                'sparse_depth.p_world').to(device)
            sparse_depth['depth_gt'] = data.get('sparse_depth.d').to(device)
            sparse_depth['camera_mat'] = data.get(
                'sparse_depth.camera_mat').to(device)
            sparse_depth['world_mat'] = data.get(
                'sparse_depth.world_mat').to(device)
            sparse_depth['scale_mat'] = data.get(
                'sparse_depth.scale_mat').to(device)
        else:
            sparse_depth = None

        return (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
                inputs, sparse_depth)

    def calc_occupancy_loss(self, logits_hat, mask_occupancy, reduction_method,
                            loss={}):
        ''' Calculates the occupancy loss.

        Args:
            logits_hat (tensor): predicted occupancy in logits
            mask_occupancy (tensor): mask for occupancy loss
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
        '''
        batch_size = logits_hat.shape[0]

        loss_occupancy = losses.occupancy_loss(
            logits_hat[mask_occupancy], reduction_method=reduction_method) * \
            self.lambda_occupied / batch_size
        loss['loss'] += loss_occupancy
        loss['loss_occupied'] = loss_occupancy

    def calc_freespace_loss(self, logits_hat, mask_freespace, reduction_method,
                            loss={}):
        ''' Calculates the freespace loss.

        Args:
            logits_hat (tensor): predicted occupancy in logits
            mask_freespace (tensor): mask for freespace loss
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
        '''
        batch_size = logits_hat.shape[0]

        loss_freespace = losses.freespace_loss(
            logits_hat[mask_freespace], reduction_method=reduction_method) * \
            self.lambda_freespace / batch_size
        loss['loss'] += loss_freespace
        loss['loss_freespace'] = loss_freespace

    def calc_depth_loss(self, mask_depth, depth_img, pixels,
                        camera_mat, world_mat, scale_mat, p_world_hat,
                        reduction_method, loss={}, eval_mode=False):
        ''' Calculates the depth loss.

        Args:
            mask_depth (tensor): mask for depth loss
            depth_img (tensor): depth image
            pixels (tensor): sampled pixels in range [-1, 1]
            camera_mat (tensor): camera matrix
            world_mat (tensor): world matrix
            scale_mat (tensor): scale matrix
            p_world_hat (tensor): predicted world points
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
            eval_mode (bool): whether to use eval mode
        '''
        if self.lambda_depth != 0 and mask_depth.sum() > 0:
            batch_size, n_pts, _ = p_world_hat.shape
            loss_depth_val = torch.tensor(10)
            # For depth values, we have to check again if all values are valid
            # as we potentially train with sparse depth maps
            depth_gt, mask_gt_depth = get_tensor_values(
                depth_img, pixels, squeeze_channel_dim=True, with_mask=True)
            mask_depth &= mask_gt_depth
            if self.depth_loss_on_world_points:
                # Applying L2 loss on world points results in the same as
                # applying L1 on the depth values with scaling (see Sup. Mat.)
                p_world = transform_to_world(
                    pixels, depth_gt.unsqueeze(-1), camera_mat, world_mat,
                    scale_mat)
                loss_depth = losses.l2_loss(
                    p_world_hat[mask_depth], p_world[mask_depth],
                    reduction_method) * self.lambda_depth / batch_size
                if eval_mode:
                    loss_depth_val = losses.l2_loss(
                        p_world_hat[mask_depth], p_world[mask_depth],
                        'mean') * self.lambda_depth
            else:
                d_pred = transform_to_camera_space(
                    p_world_hat, camera_mat, world_mat, scale_mat)[:, :, -1]
                loss_depth = losses.l1_loss(
                    d_pred[mask_depth], depth_gt[mask_depth],
                    reduction_method, feat_dim=False) * \
                    self.lambda_depth / batch_size
                if eval_mode:
                    loss_depth_val = losses.l1_loss(
                        d_pred[mask_depth], depth_gt[mask_depth],
                        'mean', feat_dim=False) * self.lambda_depth

            loss['loss'] += loss_depth
            loss['loss_depth'] = loss_depth
            if eval_mode:
                loss['loss_depth_eval'] = loss_depth_val

    def calc_photoconsistency_loss(self, mask_rgb, rgb_pred, img, pixels,
                                   reduction_method, loss, patch_size,
                                   eval_mode=False):
        ''' Calculates the photo-consistency loss.

        Args:
            mask_rgb (tensor): mask for photo-consistency loss
            rgb_pred (tensor): predicted rgb color values
            img (tensor): GT image
            pixels (tensor): sampled pixels in range [-1, 1]
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
            patch_size (int): size of sampled patch
            eval_mode (bool): whether to use eval mode
        '''
        if self.lambda_rgb != 0 and mask_rgb.sum() > 0:
            batch_size, n_pts, _ = rgb_pred.shape
            loss_rgb_eval = torch.tensor(3)
            # Get GT RGB values
            rgb_gt = get_tensor_values(img, pixels)

            # 3.1) Calculate RGB Loss
            loss_rgb = losses.l1_loss(
                rgb_pred[mask_rgb], rgb_gt[mask_rgb],
                reduction_method) * self.lambda_rgb / batch_size
            loss['loss'] += loss_rgb
            loss['loss_rgb'] = loss_rgb
            if eval_mode:
                loss_rgb_eval = losses.l1_loss(
                    rgb_pred[mask_rgb], rgb_gt[mask_rgb], 'mean') * \
                    self.lambda_rgb

            # 3.2) Image Gradient loss
            if self.lambda_image_gradients != 0:
                assert(patch_size > 1)
                loss_grad = losses.image_gradient_loss(
                    rgb_pred, rgb_gt, mask_rgb, patch_size,
                    reduction_method) * \
                    self.lambda_image_gradients / batch_size
                loss['loss'] += loss_grad
                loss['loss_image_gradient'] = loss_grad
            if eval_mode:
                loss['loss_rgb_eval'] = loss_rgb_eval

    def calc_normal_loss(self, normals, batch_size, loss={}, eval_mode=False):
        ''' Calculates the normal loss.

        Args:
            normals (list): 2 tensors (normals of surface points and of a
                randomly sampled neighbor)
            batch_size (int): batch size
            loss (dict): loss dictionary
            eval_mode (bool): whether to use eval mode
        '''
        if self.lambda_normal != 0:
            normal_loss = torch.norm(normals[0] - normals[1], dim=-1).sum() *\
                self.lambda_normal / batch_size
            loss['loss'] += normal_loss
            loss['normal_loss'] = normal_loss
            if eval_mode:
                normal_loss_eval = torch.norm(
                    normals[0] - normals[1], dim=-1).mean() * \
                    self.lambda_normal
                loss['normal_loss_eval'] = normal_loss_eval

    def calc_mask_intersection(self, mask_gt, mask_pred, loss={}):
        ''' Calculates th intersection and IoU of provided mask tensors.

        Args:
            mask_gt (tensor): GT mask
            mask_pred (tensor): predicted mask
            loss (dict): loss dictionary
        '''
        mask_intersection = (mask_gt == mask_pred).float().mean()
        mask_iou = compute_iou(
            mask_gt.cpu().float(), mask_pred.cpu().float()).mean()
        loss['mask_intersection'] = mask_intersection
        loss['mask_iou'] = mask_iou

    def compute_loss(self, data, c_clip, condition1, text,   eval_mode=False, it=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        loss = {}
        n_points = self.n_eval_points if eval_mode else self.n_training_points
        # Process data dictionary
        (img, mask_img, depth_img, world_mat, camera_mat, scale_mat,
         inputs, sparse_depth) = self.process_data_dict(data)
        #print (world_mat.shape, camera_mat.shape, scale_mat.shape)
         
         

        # Shortcuts
        device = self.device

        batch_size=1



        size=90
        pixels = arange_pixels((size,size), batch_size)[1].to(device)

        
        np.save('out/'+text+'/c.npy', c_clip.detach().cpu().numpy())

        
        
        
        #np.save('c_dvr.npy', c_dvr.detach().cpu().numpy())
        #print (torch.unique(c_dvr))
        

        
        
        loss['loss']=torch.zeros((1)).cuda() #clip_loss


        camera_mat=torch.from_numpy(np.load('../../ShapeNet/camera_mat.npy')).cuda().float().unsqueeze(0)
        scale_mat=torch.from_numpy(np.eye(4)).cuda().float().unsqueeze(0)

        #print ('clip', torch.unique(c_clip))
        c_clip_gene=self.model.generator(c_clip)
        #print ('map', torch.unique(c_clip_gene))


        N=1
        pixels=pixels.repeat(N,1,1)
        camera_mat=camera_mat.repeat(N,1,1)
        scale_mat=scale_mat.repeat(N,1,1)
        c_clip_gene=c_clip_gene.repeat(N,1)        
        for rrr in range(int(10/N)):

        
        
          world_mat=torch.zeros((N, 4,4)).cuda()
          for nn in range(N):
            world_m=torch.from_numpy(np.load(random.choice(glob.glob('../../cameras/*.npy')))).cuda().float().unsqueeze(0)
            world_m[:,:3,3]*=1.5
            #if random.uniform(0, 1)>0.9:
            #  world_m=-world_m
            world_mat[nn,:,:]=world_m
          
            
          



          
          
          
          import time
          #start=time.time()
          p_world, mask_pred, mask_zero_occupied = \
              self.model.pixels_to_world(pixels, camera_mat,
                                   world_mat, scale_mat, c_clip_gene, it)
          rgb_pred = self.model.decode_color(p_world, c=c_clip_gene)
          #print ('1', rgb_pred.shape)
          
          #mask_pred=mask_pred.int()
          rgb_pred_reshape=torch.reshape(rgb_pred,(-1,size, size,3))
          rgb_pred_reshape=torch.transpose(rgb_pred_reshape, 1,2)
          

          if rrr==0:
            rgb_pred_np_reshape=rgb_pred_reshape.detach().cpu().numpy()
            cv2.imwrite('out/'+text+'/rgbclip'+str(len(glob.glob('out/'+text+'/rgbclip*.png'))+1)+'.png',rgb_pred_np_reshape[0,:,:,::-1]*255)
          
          
          rgb_pred_nchw=(rgb_pred_reshape.permute(0,3,1,2)/1.0-0.45)/0.27
          rgb_pred_224=torch.nn.functional.interpolate(rgb_pred_nchw, (224,224))
          
  

          image_features = self.model.clip_model.encode_image(rgb_pred_224.transpose(2,3))
          clip_loss1=torch.sum((1-self.clip_criterion(image_features, condition1.detach())))*4


          loss['loss']+=clip_loss1
          

          
          mask_pred_int=mask_pred.int()
          mask_pred3=torch.unsqueeze(mask_pred_int, -1).repeat(1,1,3)
          mask_255=torch.ones(mask_pred3.shape).cuda()
          mask_pred_inv=mask_255-mask_pred3
          mask_255_loss=torch.mean(torch.abs(mask_255*mask_pred_inv-rgb_pred*mask_pred_inv))*3
          loss['loss']+= mask_255_loss



        if eval_mode:
            self.calc_mask_intersection(mask_gt, mask_pred, loss)
        
        if eval_mode:
          return loss #if eval_mode
        else:
          return loss['loss'], clip_loss1, clip_loss1, clip_loss1 


        #return loss if eval_mode else loss['loss']
          

    def calc_sparse_depth_loss(self, sparse_depth, p_world_hat, mask_pred,
                               reduction_method, loss={}, eval_mode=False):
        ''' Calculates the sparse depth loss.

        Args:
            sparse_depth (dict): dictionary for sparse depth loss calculation
            p_world_hat (tensor): predicted world points
            mask_pred (tensor): mask for predicted values
            reduction_method (string): how to reduce the loss tensor
            loss (dict): loss dictionary
            eval_mode (bool): whether to use eval mode
        '''
        if self.lambda_sparse_depth != 0:
            p_world = sparse_depth['p_world']
            depth_gt = sparse_depth['depth_gt']
            camera_mat = sparse_depth['camera_mat']
            world_mat = sparse_depth['world_mat']
            scale_mat = sparse_depth['scale_mat']

            # Shortscuts
            batch_size, n_points, _ = p_world.shape
            if self.depth_loss_on_world_points:
                loss_sparse_depth = losses.l2_loss(
                    p_world_hat[mask_pred], p_world[mask_pred],
                    reduction_method) * self.lambda_sparse_depth / batch_size
            else:
                d_pred_cam = transform_to_camera_space(
                    p_world_hat, camera_mat, world_mat, scale_mat)[:, :, -1]
                loss_sparse_depth = losses.l1_loss(
                    d_pred_cam[mask_pred], depth_gt[mask_pred],
                    reduction_method, feat_dim=False) * \
                    self.lambda_sparse_depth / batch_size

            if eval_mode:
                if self.depth_loss_on_world_points:
                    loss_sparse_depth_val = losses.l2_loss(
                        p_world_hat[mask_pred], p_world[mask_pred], 'mean') * \
                        self.lambda_sparse_depth
                else:
                    d_pred_cam = transform_to_camera_space(
                        p_world_hat, camera_mat, world_mat,
                        scale_mat)[:, :, -1]
                    loss_sparse_depth_val = losses.l1_loss(
                        d_pred_cam[mask_pred], depth_gt[mask_pred], 'mean',
                        feat_dim=False) * self.lambda_sparse_depth
                loss['loss_sparse_depth_val'] = loss_sparse_depth_val

            loss['loss'] += loss_sparse_depth
            loss['loss_sparse_depth'] = loss_sparse_depth

    def visualize(self, data, it=0, vis_type='mesh'):
        ''' Visualized the data.

        Args:
            data (dict): data dictionary
            it (int): training iteration
            vis_type (string): visualization type
        '''
        if self.multi_gpu:
            print(
                "Sorry, visualizations currently not implemented when using \
                multi GPU training.")
            return 0

        device = self.device
        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        batch_size = inputs.shape[0]
        c = self.model.encode_inputs(inputs)
        if vis_type == 'voxel':
            shape = (32, 32, 32)
            p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
            p = p.unsqueeze(0).repeat(batch_size, 1, 1)
            with torch.no_grad():
                p_r = self.model.decode(p, c=c).probs
            voxels_out = (p_r >= self.threshold).cpu().numpy()
            voxels_out = voxels_out.reshape(batch_size, 32, 32, 32)
            for i in range(batch_size):
                out_file = os.path.join(self.vis_dir, '%03d.png' % i)
                vis.visualize_voxels(voxels_out[i], out_file)
        elif vis_type == 'pointcloud':
            p = torch.rand(batch_size, 60000, 3).to(device) - 0.5
            with torch.no_grad():

                occ = self.model.decode(p, c=c).probs
                mask = occ > self.threshold

            for i in range(batch_size):
                pi = p[i][mask[i]].cpu()
                out_file = os.path.join(self.vis_dir, '%03d.png' % i)
                vis.visualize_pointcloud(pi, out_file=out_file)
        elif vis_type == 'mesh':
            try:
                mesh_list = self.generator.generate_meshes(
                    data, return_stats=False)
                for i, mesh in tqdm(enumerate(mesh_list)):
                    if self.overwrite_visualization:
                        ending = ''
                    else:
                        ending = '_%010d' % it
                    mesh_out_file = os.path.join(
                        self.vis_dir, '%03d%s.ply' % (i, ending))
                    mesh.export(mesh_out_file)
            except Exception as e:
                print("Exception occurred during visualization: ", e)
        else:
            print('The visualization type %s is not valid!' % vis_type)
