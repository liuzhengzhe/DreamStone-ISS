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
import trimesh,glob,cv2
from plyfile import PlyData,PlyElement
from im2mesh.common import (
    check_weights, get_tensor_values, transform_to_world,
    transform_to_camera_space, sample_patch_points, arange_pixels,
    make_3d_grid, compute_iou, get_occupancy_loss_points,
    get_freespace_loss_points
)

def write_ply_point(name, vertices):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	fout.close()

def write_ply_point_normal(name, vertices,  colors):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("property uchar red\n")
	fout.write("property uchar green\n")
	fout.write("property uchar blue\n")
	fout.write("end_header\n")
	if 1:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+" "+str(min(255,int(255*colors[ii,0])))+" "+str(min(255,int(255*colors[ii,1])))+" "+str(min(255,int(255*colors[ii,2])))+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract meshes from occupancy process.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--upsampling-steps', type=int, default=-1,
                        help='Overrites the default upsampling steps in config')
    parser.add_argument('--refinement-step', type=int, default=-1,
                        help='Overrites the default refinement steps in config')
    parser.add_argument('--model', type=str, default='../stage2/out/a wooden boat/model20.pt',
                        help='The initialization model path')     

    args = parser.parse_args()

    root='/'.join(args.model.split('/')[:-1])


    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # Overwrite upsamping and refinement step if desired
    if args.upsampling_steps != -1:
        cfg['generation']['upsampling_steps'] = args.upsampling_steps
    if args.refinement_step != -1:
        cfg['generation']['refinement_step'] = args.refinement_step

    # Shortcuts
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    batch_size = cfg['generation']['batch_size']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    mesh_extension = cfg['generation']['mesh_extension']

    # Dataset
    # This is for DTU when we parallelise over images
    # we do not want to treat different images from same object as
    # different objects
    cfg['data']['split_model_for_images'] = False
    dataset = config.get_dataset(cfg, mode='test', return_idx=True)

    # Model
    model = config.get_model(cfg, device=device, len_dataset=len(dataset))

    checkpoint_io = CheckpointIO('out/single_view_reconstruction/multi_view_supervision/ours_combined/model.pt', model=model)
    checkpoint_io.load(cfg['test']['model_file'], device=device)
    
    # Generator
    generator = config.get_generator(model, cfg, device=device)

    torch.manual_seed(0)
    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=True)

    # Statistics
    time_dicts = []
    vis_file_dict = {}

    # Generate
    model.eval()

    # Count how many models already created
    model_counter = defaultdict(int)
    for it, data in enumerate(tqdm(test_loader)):
        # Output folders
        mesh_dir = os.path.join(generation_dir, 'meshes')
        in_dir = os.path.join(generation_dir, 'input')
        generation_vis_dir = os.path.join(generation_dir, 'vis', )
        generation_paper_vis_dir = os.path.join(
            generation_dir, 'visualizations_paper')

        # Get index etc.
        idx = data['idx'].item()
        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'na', 'category_id': 0000}

        modelname = model_dict['model']
        category = model_dict.get('category', 'na')
        category_id = model_dict.get('category_id', 0000)
        mesh_dir = os.path.join(mesh_dir, category)
        in_dir = os.path.join(in_dir, category)
        generation_vis_dir = os.path.join(generation_vis_dir, category)
        generation_paper_vis_dir = os.path.join(generation_paper_vis_dir, category)

        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)
        if not os.path.exists(generation_vis_dir) and vis_n_outputs > 0:
            os.makedirs(generation_vis_dir)

        # Timing dict
        time_dict = {
            'idx': idx,
            'class_name': category,
            'class_id': category_id,
            'modelname': modelname,
        }
        time_dicts.append(time_dict)

        # Generate outputs
        out_file_dict = {}

        # add empty list to vis_out_file for this category
        if category not in vis_file_dict.keys():
            vis_file_dict[category] = []

        if 1:
            text='out'
            try:
              os.mkdir(text)
            except:
              pass
            t0 = time.time()
            out = generator.generate_mesh(data, root)
            time_dict['mesh'] = time.time() - t0
            # Get statistics
            try:
                mesh, stats_dict,c,pc1,pc2 = out
            except TypeError:
                mesh, stats_dict,c,pc1,pc2 = out, {}
            time_dict.update(stats_dict)



            it=9999999
            size=100
            pixels = arange_pixels((size,size), batch_size)[1].to(device)
            
            camera_mat=torch.from_numpy(np.load('../../ShapeNet/camera_mat.npy')).cuda().float().unsqueeze(0)
            scale_mat=torch.from_numpy(np.eye(4)).cuda().float().unsqueeze(0)
            
            

            patch=10
            psize=int(size/patch)



            rgb_pred=torch.zeros((size,size,3))
            mask_pred=torch.zeros((size,size,3))
        
            for rrr in range(5):
              for i in range(patch):
                world_mat=torch.from_numpy(np.load(glob.glob('../../cameras/*.npy')[rrr])).cuda().float().unsqueeze(0)
                #world_mat[:,:3,3]*=1.5
                p_world, mask_p, mask_zero_occupied = \
                    model.pixels_to_world(pixels[:,i*psize*size:(i+1)*psize*size,:], camera_mat,
                                         world_mat, scale_mat, c, it)
                rgb_p = model.decode_color(p_world, c=c)
                
                #mask_p=mask_pred.int()
                #print (rgb_pred[:,i*psize:(i+1)*psize,:].shape, rgb_p.shape)
                rgb_pred[i*psize:(i+1)*psize,:,:]=torch.reshape(rgb_p,(psize,size,3))

                mask_pred[i*psize:(i+1)*psize,:,0]=torch.reshape(mask_p,(psize,size))      
                mask_pred[i*psize:(i+1)*psize,:,1]=torch.reshape(mask_p,(psize,size))     
                mask_pred[i*psize:(i+1)*psize,:,2]=torch.reshape(mask_p,(psize,size))  
                                           
              #rgb_pred_reshape=torch.reshape(rgb_pred,(-1,size, size,3))
              #rgb_pred_reshape=torch.reshape(rgb_pred,(-1,size, size,3))
              rgb_pred=torch.transpose(rgb_pred, 0,1)
              rgb_pred_np_reshape=rgb_pred.detach().cpu().numpy()

              mask_pred=torch.transpose(mask_pred, 0,1)
              mask_pred_np_reshape=mask_pred.detach().cpu().numpy()
              rgb_pred_np_reshape[np.where(mask_pred_np_reshape==0)]=1


              
              cv2.imwrite(text+'/render'+str(rrr)+'.png',rgb_pred_np_reshape[:,:,::-1]*255)
        

            #np.save('ft1/'+text.split('/')[-1]+'.npy',c.detach().cpu().numpy())
            #continue
            write_ply_point_normal(text+"/newpc.ply", pc1, pc2/255.0)

            # Write output
            #mesh_out_file = os.path.join(
            #    mesh_dir, '%s.%s' % ("new"+modelname, mesh_extension))
            mesh.export('out/mesh.ply')

            

        #except RuntimeError:
        #    print("Error generating mesh %s (%s)." % (modelname, category))

        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category]
        if c_it < vis_n_outputs:
            # add model to vis_out_file
            vis_file_dict[str(category)].append(modelname)
            # Save output files
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                        % (c_it, k, ext))
                if cfg['data']['dataset_name'] == 'DTU':
                    # rotate for DTU to visualization purposes
                    r = R.from_euler('xz', [-90, 10], degrees=True).as_matrix() @ \
                        R.from_euler('xzy', [220, 44.9, 10.6], degrees=True
                                    ).as_matrix()
                    transform = np.eye(4).astype(np.float32)
                    transform[:3, :3] = r.astype(np.float32)
                    mesh = transform_mesh(mesh, transform)
                    mesh.export(out_file)
                else:
                    shutil.copyfile(filepath, out_file)

            if cfg['data']['input_type'] == 'image':
                img = data.get('inputs')[0].permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                out_file = os.path.join(generation_vis_dir, '%02d_input.jpg'
                                        % (c_it))
                img.save(out_file)

        model_counter[category] += 1

    # Create pandas dataframe and save
    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle(out_time_file)

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class_name']).mean()
    time_df_class.to_pickle(out_time_file_class)

    # Print results
    time_df_class.loc['mean'] = time_df_class.mean()
    print('Timings [s]:')
    print(time_df_class)

    # save vis_out_file
    vis_file_dict_name = os.path.join(generation_dir, 'vis', 'visualization_files')
    np.save(vis_file_dict_name, vis_file_dict)
