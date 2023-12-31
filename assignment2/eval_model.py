import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes, cubify
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
import numpy as np
import imageio
from starter.utils import *
import warnings

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        try:
            pred_points = sample_points_from_meshes(mesh_src, args.n_points)
            pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
        except:
            return None
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)
        if (step % args.vis_freq) == 0:
            # visualization block
            visualise_model(step, args, images_gt, mesh_gt, predictions, feed_dict)
            # pass
      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        if metrics is not None:
            f1_05 = metrics['F1@0.050000']
            if f1_05 > 90:
                visualise_model(step, args, images_gt, mesh_gt, predictions, feed_dict)
            avg_f1_score_05.append(f1_05)
            avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
            avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
            avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
        else:
            pass

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

def visualise_model(step, args, images_gt, mesh_gt, predictions, feed_dict):

    if args.type == 'vox':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mesh_src = cubify(predictions[0], thresh=0.5, device=device)
        mesh_tgt = cubify(feed_dict["voxels"][0].to(args.device), thresh=0.5, device=device)
        
        renderer = get_mesh_renderer(image_size=137, device=device)

        vertices_src = mesh_src.verts_list()[0].unsqueeze(0)
        vertices_tgt = mesh_tgt.verts_list()[0].unsqueeze(0)

        textures_src = torch.ones_like(vertices_src)
        textures_tgt = torch.ones_like(vertices_tgt)

        if textures_src.shape != torch.Size([1, 0]):
            mesh_src.textures = pytorch3d.renderer.TexturesVertex(textures_src)
            mesh_tgt.textures = pytorch3d.renderer.TexturesVertex(textures_tgt)

            gif_output_path = f"results/eval_vox_vis_{step}.gif"
            rgb_output_path = f"results/eval_vox_vis_{step}.png"

            dist = 3 # distance of camera form origin
            elevation = torch.tensor([0.0])  # Elevation angle
            azimuth = torch.linspace(0, 2 * np.pi, 36) # angle along reference plane
            images = []

            lights = pytorch3d.renderer.PointLights(location=[[0, 0, dist]], device=device)

            images_gt = images_gt.detach().cpu().numpy()[0]
            plt.imsave(rgb_output_path,images_gt)
            for azm in azimuth:
            
                rot,tr = pytorch3d.renderer.cameras.look_at_view_transform(dist, elevation, azm, degrees = False)

                # Set the camera's rotation matrix for the current view
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                    R=rot, T=tr, fov=60, device=device
                )        

                rend_src = renderer(mesh_src, cameras=cameras, lights=lights)
                rend_src = rend_src[0, ..., :3].detach().cpu().numpy() 
                rend_uint8_src = (rend_src * 255).clip(0, 255).astype(np.uint8)

                rend_tgt = renderer(mesh_tgt, cameras=cameras, lights=lights)
                rend_tgt = rend_tgt[0, ..., :3].detach().cpu().numpy() 
                rend_uint8_tgt = (rend_tgt * 255).clip(0, 255).astype(np.uint8)

                stacked = np.hstack((rend_uint8_src, rend_uint8_tgt))
                images.append(stacked)

            imageio.mimsave(gif_output_path, images, duration=int(1000/15), loop=0)
        else:
            pass

    if args.type == 'point':
        images_list = []
        pred_points = predictions.detach().cpu().numpy()
        gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
        xs = pred_points[0, :, 0]; ys = pred_points[0, :, 1]; zs = pred_points[0, :, 2]
        xt = gt_points[0, :, 0]; yt = gt_points[0, :, 1]; zt = gt_points[0, :, 2]
        
        images_gt = images_gt.detach().cpu().numpy()[0]

        for azim in range (0,360,10):
            fig = plt.figure()
        
            ax1 = fig.add_subplot(131)  # Bottom-left subplot for the RGB image
            ax1.imshow(images_gt)
            ax1.grid(False)
            ax1.axis('off')

            ax2 = fig.add_subplot(132, projection = '3d')
            ax2.view_init(elev=10, azim=azim)
            ax2.scatter(xs,zs,ys, c=(1, 0, 0), marker = '.')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('z')
            ax2.grid(False)
            ax2.axis('off')

            ax3 = fig.add_subplot(133, projection = '3d')
            ax3.view_init(elev=10, azim=azim)
            ax3.scatter(xt,zt,yt, c=(0, 1, 0), marker = '.')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_zlabel('z')
            ax3.grid(False)
            ax3.axis('off')

            plt.tight_layout()

            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())

            images_list.append(frame)
            plt.close()

        imageio.mimsave(f"eval_pc_vis_n1000_{step}.gif", images_list, duration = (1000/15), loop = 0)


    if args.type == 'mesh':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mesh_src = predictions
        mesh_tgt = feed_dict["mesh"].to(args.device)
        renderer = get_mesh_renderer(image_size=256, device=device)

        vertices_src = mesh_src.verts_list()[0].unsqueeze(0)
        vertices_tgt = mesh_tgt.verts_list()[0].unsqueeze(0)

        textures_src = torch.ones_like(vertices_src) * torch.tensor([0.8, 0.2, 0.6]).to(device)
        textures_tgt = torch.ones_like(vertices_tgt) * torch.tensor([0.1, 0.9, 0.3]).to(device)

        mesh_src.textures = pytorch3d.renderer.TexturesVertex(textures_src)
        mesh_tgt.textures = pytorch3d.renderer.TexturesVertex(textures_tgt)

        gif_output_path = f"eval_mesh_vis_{step}.gif"
        rgb_output_path = f"eval_mesh_vis_{step}.png"

        dist = 1 # distance of camera form origin
        elevation = torch.tensor([0.0])  # Elevation angle
        azimuth = torch.linspace(0, 2 * np.pi, 36) # angle along reference plane
        images = []

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, dist]], device=device)

        images_gt = images_gt.detach().cpu().numpy()[0]
        plt.imsave(rgb_output_path,images_gt)
        for azm in azimuth:
        
            rot,tr = pytorch3d.renderer.cameras.look_at_view_transform(dist, elevation, azm, degrees = False)

            # Set the camera's rotation matrix for the current view
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=rot, T=tr, fov=60, device=device
            )        

            rend_src = renderer(mesh_src, cameras=cameras, lights=lights)
            rend_src = rend_src[0, ..., :3].detach().cpu().numpy() 
            rend_uint8_src = (rend_src * 255).clip(0, 255).astype(np.uint8)

            rend_tgt = renderer(mesh_tgt, cameras=cameras, lights=lights)
            rend_tgt = rend_tgt[0, ..., :3].detach().cpu().numpy() 
            rend_uint8_tgt = (rend_tgt * 255).clip(0, 255).astype(np.uint8)

            # stacked = np.hstack((rend_uint8_src, rend_uint8_tgt))
            images.append(rend_uint8_src)

        imageio.mimsave(gif_output_path, images, duration=int(1000/15), loop=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)