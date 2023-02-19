# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Main training loop."""

import os
import copy
import json
import numpy as np
import torch
import dnnlib
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics import metric_main
import nvdiffrast.torch as dr
import time
from training.inference_utils import save_image_grid, save_visualization


# ----------------------------------------------------------------------------
# Function to save the real image for discriminator training
def setup_snapshot_image_grid(training_set, random_seed=0, inference=False):
    rnd = np.random.RandomState(random_seed)
    grid_w = 7
    grid_h = 4
    min_w = 8 if inference else grid_w
    min_h = 9 if inference else grid_h
    gw = np.clip(1024 // training_set.image_shape[2], min_w, 32)
    gh = np.clip(1024 // training_set.image_shape[1], min_h, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, masks = zip(*[training_set[i][:3] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels), masks


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    # We use this function to remove or change custom kwargs for dataset
    # we used these kwargs to comput md5 for the cache file of FID
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs



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
# ----------------------------------------------------------------------------
def training_loop(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        G_opt_kwargs={},  # Options for generator optimizer.
        D_opt_kwargs={},  # Options for discriminator optimizer.
        loss_kwargs={},  # Options for loss function.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=0.05,  # EMA ramp-up coefficient. None = no rampup.
        G_reg_interval=None,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_kimg=0,  # First kimg to report when resuming training. ######
        abort_fn=None,
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
        inference_vis=False,  # Whether running inference or not.
        detect_anomaly=False,
        resume_pretrain=None,
):
    '''from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()
    if num_gpus > 1:
        torch.distributed.barrier()
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    if rank == 0:
        print('Loading training set...')

    # Set up training dataloader
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)

    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus,
            **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    if rank == 0:
        print('Constructing networks...')

    # Constructing networks'''
    device='cuda'
    common_kwargs = dict(
        c_dim=3, img_resolution=512, img_channels=3)
    G_kwargs['device'] = device
    D_kwargs['device'] = device

    if num_gpus > 1:
        torch.distributed.barrier()
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        
        # We're not reusing the loading function from stylegan3 codebase,
        # since we have some variables that are not picklable.
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        print (model_state_dict['G'])
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        D.load_state_dict(model_state_dict['D'], strict=True)

    if rank == 0:
        print('Setting up augmentation...')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)  # Broadcast from GPU 0

    if rank == 0:
        print('Setting up training phases...')

    # Constructing loss functins and optimizer
    loss = dnnlib.util.construct_class_by_name(
        device=device, G=G, D=D, **loss_kwargs)  # subclass of training.loss.Loss
    '''phases = []

    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval),
                                                   ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(
                params=module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(
                module.parameters(),
                **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    grid_size = None
    grid_z = None
    grid_c = None

    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels, masks = setup_snapshot_image_grid(training_set=training_set, inference=inference_vis)
        masks = np.stack(masks)
        images = np.concatenate((images, masks[:, np.newaxis, :, :].repeat(3, axis=1) * 255.0), axis=-1)
        if not inference_vis:
            save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        torch.manual_seed(1234)
        grid_z = torch.randn([images.shape[0], G.z_dim], device=device).split(1)  # This one is the latent code for shape generation
        grid_c = torch.ones(images.shape[0], device=device).split(1)  # This one is not used, just for the compatiable with the code structure.

    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    if progress_fn is not None:
        progress_fn(0, total_kimg)

    optim_step = 0'''

    text='a red tall chair'
    device='cuda'
    import clip
    text = clip.tokenize(text).to(device)

    model, preprocess = clip.load("ViT-B/32", device=device)
    text_features = model.encode_text(text)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #import train_layernorm_multi_g2
    mapper = generator_mapper(4,64).cuda()
    mapper.load_state_dict(torch.load('model_chair_200.pt'), strict=True)
    mapper.eval()

    g1, g2, c=mapper(text_features.detach().float())
    g1=torch.reshape(g1, (1,11,512))
    g2=torch.reshape(g2, (1,11,512))
    c=torch.reshape(c, (1,9,512))

    #print ('bbbb', g1.shape, g2.shape, c.shape)

    gen_z=torch.cat((g1,g2),1)#[:,0,:]
    gen_c=c#[:,0,:]

    # Training Iterations
    while True:
        '''loss.accumulate_gradients(
                    phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                    gain=phase.interval, cur_nimg=cur_nimg)'''
        gen_img, gen_sdf, _gen_ws, gen_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _gen_ws_geo, \
                sdf_reg_loss, render_return_value = loss.run_lzz(
                    gen_z, gen_c, return_shape=True
                )
        #print ('genimg.shape',gen_img.shape)
        gen_img=torch.permute(gen_img[0,:3,:,:],(1,2,0))
        mask=torch.permute(gen_img[0,3:,:,:],(1,2,0))
        img=gen_img.detach().cpu().numpy()
        mask=mask.detach().cpu().numpy()
        import cv2,glob
        cv2.imwrite('img'+str(len(glob.glob('img*.png')))+'.png',(img+1)*126)
        cv2.imwrite('mask'+str(len(glob.glob('mask*.png')))+'.png',mask*255)


        continue
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c, real_mask = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1)
            real_mask = real_mask.to(device).to(torch.float32).unsqueeze(dim=1)
            real_mask = (real_mask > 0).float()
            phase_real_img = torch.cat([phase_real_img, real_mask], dim=1)
            phase_real_img = phase_real_img.split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * (batch_size // num_gpus), G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split((batch_size // num_gpus))]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in
                         range(len(phases) * (batch_size // num_gpus))]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size // num_gpus)]
        optim_step += 1
        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=False)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(
                    phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                    gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    if torch.isnan(flat).any():
                        print('==> find nan values')
                        print('==> nan grad')  # We should keep track of this for nan!!!!!!
                        for name, p in phase.module.named_parameters():
                            if p.grad is not None:
                                if torch.isnan(p.grad).any():
                                    print(name)
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if detect_anomaly:
            print('==> finished one round')

        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]

        if rank == 0:
            print(' '.join(fields))

        if detect_anomaly:
            continue

        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0) and (
                not detect_anomaly):
            with torch.no_grad():
                print('==> start visualization')
                save_visualization(
                    G_ema, grid_z, grid_c, run_dir, cur_nimg, grid_size, cur_tick,
                    image_snapshot_ticks,
                    save_all=(cur_tick % (image_snapshot_ticks * 4) == 0) and training_set.resolution < 512,
                )
                print('==> saved visualization')

            # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0) and not detect_anomaly:  # and (cur_tick != 0 or resume_pretrain is not None):  ###########
            snapshot_data = dict(
                G=G, D=D, G_ema=G_ema)
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module) and not isinstance(value, dr.ops.RasterizeGLContext):
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema|ctx)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                all_model_dict = {'G': snapshot_data['G'].state_dict(), 'G_ema': snapshot_data['G_ema'].state_dict(),
                                  'D': snapshot_data['D'].state_dict()}
                torch.save(all_model_dict, snapshot_pkl.replace('.pkl', '.pt'))

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            with torch.no_grad():
                for metric in metrics:
                    if training_set_kwargs['split'] != 'all':
                        if rank == 0:
                            print('====> use validation set')
                        training_set_kwargs['split'] = 'val'
                    training_set_kwargs = clean_training_set_kwargs_for_metrics(training_set_kwargs)
                    with torch.no_grad():
                        result_dict = metric_main.calc_metric(
                            metric=metric, G=snapshot_data['G_ema'],
                            dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)
            if rank == 0:
                print('==> finished evaluate metrics')

        # Collect statistics.
        for phase in phases:
            value = []
            with torch.no_grad():
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        ##### Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
    # Done.
    if rank == 0:
        print()
        print('Exiting...')
