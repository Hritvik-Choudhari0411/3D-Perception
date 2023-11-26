import torch
import pytorch3d
import matplotlib.pyplot as plt
import numpy as np
import imageio
import tqdm
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    PointsRasterizationSettings,
    HardPhongShader,
    AlphaCompositor
    # TextureVertex
)

from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_obj
from pytorch3d.ops import cubify 


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def render_mesh_gif(mesh, output, device=None, dist=3):
    
    rotations, translations = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=torch.linspace(0,360,36),
        device = device
      )

    image_list = []

    # Add Lights and render image.
    lights = pytorch3d.renderer.PointLights(
            location=[[0, 0, -4]],
            device=device
            )

    for i, pose in enumerate(zip(rotations,translations)):
        # Update the camera position.
        R = pose[0].unsqueeze(0)
        T = pose[1].unsqueeze(0)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )

        renderer = get_mesh_renderer(image_size=256, device=device)

        rend = renderer(mesh, cameras=cameras)

        image = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        image_list.append(image)

    # Resolve image for Imageio
    image_list = [(image * 255).astype('uint8') for image in image_list]

    imageio.mimsave(output, image_list, duration=60, loop=0)


def render_voxels(voxels, output):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print("before cubify")
    mesh = cubify(voxels, thresh=0.5, device=device)
    # print("AFTER cubify")
    vertices = mesh.verts_list()[0].unsqueeze(0)
    texture = torch.ones_like(vertices)
    texture *= torch.tensor([0.5, 0.2, 0.6], device=device)

    mesh.textures = TexturesVertex(texture)

    render_mesh_gif(mesh, output, device=device)


def get_points_renderer(image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def render_point_cloud(point_cloud_src, point_cloud_tgt, output):

    point_cloud_src = point_cloud_src.detach().cpu().numpy()
    point_cloud_tgt = point_cloud_tgt.detach().cpu().numpy()
    xs = point_cloud_src[0, :, 0];ys = point_cloud_src[0, :, 1];zs = point_cloud_src[0, :, 2]
    xt = point_cloud_tgt[0, :, 0];yt = point_cloud_tgt[0, :, 1];zt = point_cloud_tgt[0, :, 2]

    pc = []

    for azim in tqdm.tqdm(range(0,360,10)):
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection = '3d')
        ax1.view_init(elev=10, azim=azim)
        ax1.scatter(xs,zs,ys, edgecolor=(0.49, 0.23, 0.66))
        ax1.grid(False)
        ax1.axis('off')

        ax2 = fig.add_subplot(122, projection = '3d')
        ax2.view_init(elev=10, azim=azim)
        ax2.scatter(xt,zt,yt, edgecolor=(0.66, 0.23, 0.49))
        ax2.grid(False)
        ax2.axis('off')

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        pc.append(frame)
        plt.close()
    imageio.mimsave(output, pc, duration=int(1000/15), loop=0)



def render_mesh(mesh, output, dist=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    vertices = mesh.verts_list()[0].unsqueeze(0)
    texture = torch.ones_like(vertices)
    texture *= torch.tensor([0.49, 0.23, 0.66], device=device)

    mesh.textures = TexturesVertex(texture)

    render_mesh_gif(mesh, output, device=device)

def sec_2_6(point_cloud, point_cloud_tgt, output):
    # get vertices from point cloud and point cloud target
    vertices = point_cloud[0].detach().cpu().numpy()
    vertices_tgt = point_cloud_tgt[0].detach().cpu().numpy()

    # get x, y, z coordinates
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    zs = vertices[:, 2]

    xt = vertices_tgt[:, 0]
    yt = vertices_tgt[:, 1]
    zt = vertices_tgt[:, 2]

    pc = []

    for azim in tqdm.tqdm(range(0, 360, 10)):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection="3d")
        ax1.view_init(elev=10, azim=azim)
        ax1.scatter(xs, zs, ys, edgecolor=(0, 0, 1))
        ax1.scatter(xt, zt, yt, edgecolor=(0.5, 0.5, 0))
        ax1.grid(False)
        ax1.axis("off")

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())

        pc.append(frame)
        plt.close()
        imageio.mimsave(output, pc, duration=int(1000 / 15),loop=0)

