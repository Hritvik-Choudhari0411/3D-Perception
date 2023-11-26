import argparse
import os
import time
import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes, cubify 
from pytorch3d.structures import Meshes
import dataset_location
import torch
import pytorch3d
from starter.utils import *
import matplotlib.pyplot as plt
import numpy as np
import mcubes
import imageio
from scipy.spatial.transform import Rotation as R

def get_args_parser():
    # Define command-line arguments for the script.
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-4, type=float)  # Learning rate
    parser.add_argument('--max_iter', default=10000, type=int)  # Maximum number of iterations
    parser.add_argument('--log_freq', default=1000, type=int)  # Logging frequency
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)  # Type of data (voxel, point cloud, mesh)
    parser.add_argument('--n_points', default=5000, type=int)  # Number of points
    parser.add_argument('--w_chamfer', default=1.0, type=float)  # Weight for Chamfer loss
    parser.add_argument('--w_smooth', default=0.1, type=float)  # Weight for smoothness loss
    parser.add_argument('--device', default='cuda', type=str)  # Device (e.g., 'cuda' or 'cpu')
    return parser

################################################################################################

'''Optimization functions'''

def fit_mesh(mesh_src, mesh_tgt, args, device):
    # Function to fit a source mesh to a target mesh.
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    
    loss_history = []

    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        

        loss_history.append(loss_vis)

    mesh_src.offset_verts_(deform_vertices_src)

    print("Visualizing..")
    visualize_mesh(mesh_src, mesh_tgt, device=args.device, n_points=args.n_points)


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    # Function to fit a source point cloud to a target point cloud.
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    
    loss_history = []
    
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()
        loss_history.append(loss_vis)

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Visualizing..')
    gif_array = visualize_pointcloud(pointclouds_src, pointclouds_tgt)
    imageio.mimsave("results/point_cloud_visualization.gif", gif_array, duration=int(1000/15), loop=0)



def fit_voxel(voxels_src, voxels_tgt, args):
    # Function to fit a source voxel to a target voxel.
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print("Visualizing..")
    visualize_vox(voxels_src, voxels_tgt, device=args.device)
    
# Training the oprimization function
def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)
    feed = r2n2_dataset[0]
    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()

    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']

        # fitting
        fit_voxel(voxels_src, voxels_tgt, args)

    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)

        # fitting
        fit_pointcloud(pointclouds_src, pointclouds_tgt, args)        
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

        # fitting
        fit_mesh(mesh_src, mesh_tgt, args, device=args.device)  

####################################################################################################

'''Visualization functions'''

def visualize_vox(src, tgt, device):
    # Visualize voxel data.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mesh_src = cubify(src, thresh=0.5, device=device)
    mesh_tgt = cubify(tgt, thresh=0.5, device=device)

    renderer = get_mesh_renderer(image_size=512, device=args.device)

    vertices_src, faces_src = mesh_src.verts_packed(), mesh_src.faces_packed()
    vertices_tgt, faces_tgt = mesh_tgt.verts_packed(), mesh_tgt.faces_packed()

    textures_src = torch.ones_like(vertices_src) * 0.2
    textures_tgt = torch.ones_like(vertices_tgt) * 0.8


    textures_src = pytorch3d.renderer.TexturesVertex(textures_src.unsqueeze(0))
    textures_tgt = pytorch3d.renderer.TexturesVertex(textures_tgt.unsqueeze(0)) 
    mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src], textures=textures_src).to(device)
    mesh_tgt = pytorch3d.structures.Meshes([vertices_tgt], [faces_tgt], textures=textures_tgt).to(device)
    output_path = "results/voxel_visualization.gif"

    dist = 3 # distance of camera form origin
    elevation = torch.tensor([0.0])  # Elevation angle
    azimuth = torch.linspace(0, 2 * np.pi, 36) # angle along reference plane
    images = []
    
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, dist]], device=device)
    
    for angle in azimuth:
        
        R,T = pytorch3d.renderer.cameras.look_at_view_transform(dist, elevation, angle, degrees = False)

        # Set the camera's rotation matrix for the current view
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )        

        rend_src = renderer(mesh_src, cameras=cameras, lights=lights)
        rend_src = rend_src[0, ..., :3].detach().cpu().numpy() 
        rend_src = (rend_src * 255).clip(0, 255).astype(np.uint8)

        rend_tgt = renderer(mesh_tgt, cameras=cameras, lights=lights)
        rend_tgt = rend_tgt[0, ..., :3].detach().cpu().numpy() 
        rend_tgt = (rend_tgt * 255).clip(0, 255).astype(np.uint8)

        stacked = np.hstack((rend_src, rend_tgt))
        images.append(stacked)

    imageio.mimsave(output_path, images, duration= 0.5, loop=0)

def visualize_mesh(mesh_src, mesh_tgt, device, n_points):
    mesh_src = mesh_src[0]
    mesh_tgt = mesh_tgt[0]  # Assuming mesh_tgt is also a Meshes object

    renderer = get_mesh_renderer(image_size=512, device=args.device)

    min_value = -2
    max_value = 2

    # Rendering for mesh_src
    vertices_src, faces_src = mesh_src.verts_packed(), mesh_src.faces_packed()
    textures_src = torch.ones_like(vertices_src)

    textures_src = pytorch3d.renderer.TexturesVertex(textures_src.unsqueeze(0))
    mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src], textures=textures_src).to(device)

    # Rendering for mesh_tgt
    vertices_tgt, faces_tgt = mesh_tgt.verts_packed(), mesh_tgt.faces_packed()
    textures_tgt = torch.zeros_like(vertices_tgt)

    textures_tgt = pytorch3d.renderer.TexturesVertex(textures_tgt.unsqueeze(0))
    mesh_tgt = pytorch3d.structures.Meshes([vertices_tgt], [faces_tgt], textures=textures_tgt).to(device)

    num_frames = 60
    output_path = "results/mesh_visualization.gif"

    elevations = torch.linspace(0, 360, num_frames, device=device)
    azimuths = torch.linspace(0, 360, num_frames, device=device)
    images = []

    lights = pytorch3d.renderer.PointLights(location=[[3.0, 0, 0]]).to(device)
    
    for view_angle in range(0, 361, 5):
        # Compute the rotation matrix for the current view angle
        angle = torch.tensor([view_angle], dtype=torch.float32)
        cos_angle = torch.cos(angle * (2 * torch.pi / 360))  # Convert degrees to radians
        sin_angle = torch.sin(angle * (2 * torch.pi / 360))
        
        R = torch.tensor([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ], dtype=torch.float32, device=device)

        # Set the camera's rotation matrix for the current view
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.unsqueeze(0), T=torch.tensor([[0, 0, 1]], device=device), fov=60, device=device
        )

        # Render mesh_src
        rend_src = renderer(mesh_src, cameras=cameras, lights=lights)
        rend_src = rend_src[0, ..., :3].detach().cpu().numpy()

        # Render mesh_tgt
        rend_tgt = renderer(mesh_tgt, cameras=cameras, lights=lights)
        rend_tgt = rend_tgt[0, ..., :3].detach().cpu().numpy()

        # Combine the two renderings side by side
        combined_rend = np.concatenate((rend_src, rend_tgt), axis=1)

        rend_uint8 = (combined_rend * 255).clip(0, 255).astype(np.uint8)
        images.append(rend_uint8)

    imageio.mimsave(output_path, images, duration= 0.5, loop=0)


def visualize_pointcloud(src, tgt):
     # Visualize point cloud data.
    src = src[0].cpu().detach().numpy()
    tgt = tgt[0].cpu().detach().numpy()

    # Rotate the source and target point clouds to the desired orientation
    # Rotate by 180 degrees around the x-axis to make the chair upright
    rotation = R.from_euler('x', -90, degrees=True)

    src = src.dot(rotation.as_matrix())
    tgt = tgt.dot(rotation.as_matrix())

    x_src = src[:, 0]
    y_src = src[:, 1]
    z_src = src[:, 2]

    x_tgt = tgt[:, 0]
    y_tgt = tgt[:, 1]
    z_tgt = tgt[:, 2]
    frames = []

    for angle in range (0,360,10):
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection = '3d')
        ax1.view_init(elev=20, azim=angle)  # Set the viewing angle
        ax1.scatter(x_src,y_src,z_src, c= 'r', marker='.')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_title("Optimized Point Cloud")
        ax1.grid(False)
        ax1.axis('off')

        ax2 = fig.add_subplot(122, projection = '3d')
        ax2.view_init(elev=20, azim=angle)  # Set the viewing angle
        ax2.scatter(x_tgt,y_tgt,z_tgt, c= 'g', marker='.')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.set_title("Ground Truth Point Cloud")
        ax2.grid(False)
        ax2.axis('off')
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        frames.append(frame)
        plt.close(fig)
    return frames     

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)