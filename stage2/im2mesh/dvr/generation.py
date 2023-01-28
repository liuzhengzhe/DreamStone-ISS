import torch
import torch.optim as optim
from torch import autograd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import trimesh
from im2mesh.utils import libmcubes
from im2mesh.common import make_3d_grid
from im2mesh.utils.libsimplify import simplify_mesh
from im2mesh.utils.libmise import MISE
import time
from im2mesh.common import transform_pointcloud
import numpy as np 
import math
from PIL import Image
import torchvision.transforms as transforms
import glob

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()])

def sample_points_triangle(vertices, triangles, num_of_points):
	epsilon = 1e-6
	triangle_area_list = np.zeros([len(triangles)],np.float32)
	triangle_normal_list = np.zeros([len(triangles),3],np.float32)
	for i in range(len(triangles)):
		#area = |u x v|/2 = |u||v|sin(uv)/2
		a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
		x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
		ti = b*z-c*y
		tj = c*x-a*z
		tk = a*y-b*x
		area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
		if area2<epsilon:
			triangle_area_list[i] = 0
			triangle_normal_list[i,0] = 0
			triangle_normal_list[i,1] = 0
			triangle_normal_list[i,2] = 0
		else:
			triangle_area_list[i] = area2
			triangle_normal_list[i,0] = ti/area2
			triangle_normal_list[i,1] = tj/area2
			triangle_normal_list[i,2] = tk/area2
	
	triangle_area_sum = np.sum(triangle_area_list)
	sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

	triangle_index_list = np.arange(len(triangles))

	point_normal_list = np.zeros([num_of_points,6],np.float32)
	count = 0
	watchdog = 0

	while(count<num_of_points):
		np.random.shuffle(triangle_index_list)
		watchdog += 1
		if watchdog>100:
			print("infinite loop here!")
			return point_normal_list
		for i in range(len(triangle_index_list)):
			if count>=num_of_points: break
			dxb = triangle_index_list[i]
			prob = sample_prob_list[dxb]
			prob_i = int(prob)
			prob_f = prob-prob_i
			if np.random.random()<prob_f:
				prob_i += 1
			normal_direction = triangle_normal_list[dxb]
			u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
			v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
			base = vertices[triangles[dxb,0]]
			for j in range(prob_i):
				#sample a point here:
				u_x = np.random.random()
				v_y = np.random.random()
				if u_x+v_y>=1:
					u_x = 1-u_x
					v_y = 1-v_y
				ppp = u*u_x+v*v_y+base
				
				point_normal_list[count,:3] = ppp
				point_normal_list[count,3:] = normal_direction
				count += 1
				if count>=num_of_points: break

	return point_normal_list
 
 

class Generator3D(object):
    '''  Generator class for DVRs.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained DVR model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        simplify_nfaces (int): number of faces the mesh should be simplified to
        refine_max_faces (int): max number of faces which are used as batch
            size for refinement process (we added this functionality in this
            work)
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1,
                 simplify_nfaces=None, with_color=False,
                 refine_max_faces=10000):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.simplify_nfaces = simplify_nfaces
        self.with_color = with_color
        self.refine_max_faces = refine_max_faces

    def generate_mesh(self, data,  text,return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        #inputs=(inputs-0.45)/0.27
        import clip
        
        #print (self.model)
        #exit()
        #model.train()  
        #c, c_std =self.model.encode_inputs(inputs)
        #text=text.split('/')[-1]
        #print (text)
        #np.save('fg/'+text+'.npy', c_std.detach().cpu().numpy())
        #return text

        #np.save('dvr.npy', c_std.detach().cpu().numpy())
        #print (text+'/c.npy','text feature load path')
        print ('text', text,'text')
        c=torch.from_numpy(np.load(text+'/c.npy')).cuda()
        
        '''print (c,'c1')
        text=text.split('/')[-1]
        print ('text', text)
        text=clip.tokenize(text).to(device) #"ferry boat watercraft ship").to(device)
        c = self.model.clip_model.encode_text(text).float()'''

        '''from PIL import Image

        model, preprocess = clip.load("ViT-B/32", device=device)
        image_path='/mnt/sdc/lzz/ShapeNet/03636649/13f46d5ae3e33651efd0188089894554/img_choy2016/000.jpg'
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)    
        c = model.encode_image(image)'''


        #c  =  c/ c.norm(dim=-1, keepdim=True) # normalize to sphere
        #print (c,'c2')'''
        #print (c)
        c=self.model.generator(c.float())
        #np.save('feature/'+text.split('/')[-1], c.detach().cpu().numpy())
        #print (c.shape, c_std.shape)
        #c=torch.from_numpy(np.load('fg/a TV monitor.npy')).cuda()
        
        #np.save('final.npy', c.detach().cpu().numpy())
        #c_std=torch.from_numpy(np.load('fg2/a long luxury black car.npy')).cuda()
        #print ('yesssss')
        import glob

        '''paths=glob.glob('nearest2/*')
        for path in paths:
            text0=path.split('/')[-1].split('.')[0]

            text=clip.tokenize(text0).to(device) #"ferry boat watercraft ship").to(device)
            c = self.model.clip_model.encode_text(text).float()
            c  =  c/ c.norm(dim=-1, keepdim=True) 
            c=self.model.generator(c.float())
            np.save('ft1/'+text0+'.npy', c.detach().cpu().numpy())
        exit()'''
        '''paths=glob.glob('nearest4/*/000.jpg')
        for path in paths:
            image = Image.open(path).convert("RGB")
            text=path.split('/')[-2]
            image=transform(image).cuda().unsqueeze(0)

            c, c_std =self.model.encode_inputs(image)
            c=self.model.generator(c.float())
            ##print (c.shape, c_std.shape)
            np.save('fg4/'+text+'.npy', c_std.detach().cpu().numpy())
            np.save('fi4/'+text+'.npy', c.detach().cpu().numpy())

        exit()'''
        mesh,pc1,pc2 = self.generate_from_latent(c, stats_dict=stats_dict,
                                         data=data, **kwargs)

        return mesh, stats_dict, c,pc1,pc2

    def generate_meshes(self, data, return_stats=True):
        ''' Generates the output meshes with data of batch size >=1

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)

        meshes = []
        for i in range(inputs.shape[0]):
            input_i = inputs[i].unsqueeze(0)
            c = self.model.encode_inputs(input_i)
            mesh = self.generate_from_latent(c, stats_dict=stats_dict)
            meshes.append(mesh)

        return meshes

    def generate_pointcloud(self, mesh, data=None, n_points=2000000,
                            scale_back=True):
        ''' Generates a point cloud from the mesh.

        Args:
            mesh (trimesh): mesh
            data (dict): data dictionary
            n_points (int): number of point cloud points
            scale_back (bool): whether to undo scaling (requires a scale
                matrix in data dictionary)
        '''
        pcl = mesh.sample(n_points).astype(np.float32)

        if scale_back:
            scale_mat = data.get('camera.scale_mat_0', None)
            if scale_mat is not None:
                pcl = transform_pointcloud(pcl, scale_mat[0])
            else:
                print('Warning: No scale_mat found!')
        pcl_out = trimesh.Trimesh(vertices=pcl, process=False)
        return pcl_out

    def generate_from_latent(self, c=None, stats_dict={}, data=None,
                             **kwargs):
        ''' Generates mesh from latent.

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, c, **kwargs).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh,pc1,pc2 = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh,pc1,pc2

    def eval_points(self, p, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model.decode(pi, c, **kwargs).logits
                


            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)



        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        # mesh_pymesh = pymesh.form_mesh(vertices, triangles)
        # mesh_pymesh = fix_pymesh(mesh_pymesh)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0
        else:
            normals = None
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               # vertex_colors=vertex_colors,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        # Estimate Vertex Colors
        if self.with_color and not vertices.shape[0] == 0:
            t0 = time.time()
            vertex_colors = self.estimate_colors(np.array(mesh.vertices), c)
            stats_dict['time (color)'] = time.time() - t0
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=vertex_colors, process=False)





        
        sampled_points_normals = sample_points_triangle(vertices, triangles, 2048)
        vertices_tensor=torch.from_numpy(vertices.astype(np.float32)).cuda()
        #sampled_points_normals_int=sampled_points_normals#.astype('int')    
            
        sampled_colors = self.estimate_colors(np.array(sampled_points_normals[:,:3]), c)

        return mesh,sampled_points_normals, sampled_colors

    def estimate_colors(self, vertices, c=None):
        ''' Estimates vertex colors by evaluating the texture field.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)
        colors = []
        for vi in vertices_split:
            vi = vi.to(device)
            with torch.no_grad():
                ci = self.model.decode_color(
                    vi.unsqueeze(0), c).squeeze(0).cpu()
            colors.append(ci)

        colors = np.concatenate(colors, axis=0)
        colors = np.clip(colors, 0, 1)
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([
            colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)],
            axis=1)
        return colors

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces)

        # detach c; otherwise graph needs to be retained
        # caused by new Pytorch version?
        c = c.detach()

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-5)

        # Dataset
        ds_faces = TensorDataset(faces)
        dataloader = DataLoader(ds_faces, batch_size=self.refine_max_faces,
                                shuffle=True)

        # We updated the refinement algorithm to subsample faces; this is
        # usefull when using a high extraction resolution / when working on
        # small GPUs
        it_r = 0
        while it_r < self.refinement_step:
            for f_it in dataloader:
                f_it = f_it[0].to(self.device)
                optimizer.zero_grad()

                # Loss
                face_vertex = v[f_it]
                eps = np.random.dirichlet((0.5, 0.5, 0.5), size=f_it.shape[0])
                eps = torch.FloatTensor(eps).to(self.device)
                face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

                face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
                face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
                face_normal = torch.cross(face_v1, face_v2)
                face_normal = face_normal / \
                    (face_normal.norm(dim=1, keepdim=True) + 1e-10)

                face_value = torch.cat([
                    torch.sigmoid(self.model.decode(p_split, c).logits)
                    for p_split in torch.split(
                        face_point.unsqueeze(0), 20000, dim=1)], dim=1)

                normal_target = -autograd.grad(
                    [face_value.sum()], [face_point], create_graph=True)[0]

                normal_target = \
                    normal_target / \
                    (normal_target.norm(dim=1, keepdim=True) + 1e-10)
                loss_target = (face_value - threshold).pow(2).mean()
                loss_normal = \
                    (face_normal - normal_target).pow(2).sum(dim=1).mean()

                loss = loss_target + 0.01 * loss_normal

                # Update
                loss.backward()
                optimizer.step()

                # Update it_r
                it_r += 1

                if it_r >= self.refinement_step:
                    break

        mesh.vertices = v.data.cpu().numpy()
        return mesh

