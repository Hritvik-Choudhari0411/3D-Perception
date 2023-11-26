import torch
from pytorch3d.loss import mesh_laplacian_smoothing as mls

# Define losses

# Voxel loss computes the Binary Cross Entropy (BCE) with logits loss between two voxel grids.
def voxel_loss(voxel_src, voxel_tgt):
    loss = torch.nn.BCEWithLogitsLoss()(voxel_src, voxel_tgt)
    return loss

# Chamfer loss measures the dissimilarity between two point clouds.
def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_tgt: b x n_points x 3

    # Implement Chamfer loss from scratch

    batch_size, n_points_src, _ = point_cloud_src.size()
    n_points_tgt = point_cloud_tgt.size(1)

    # Calculate the distance between all pairs of points in the source and target clouds.
    dist_src_to_tgt = torch.cdist(point_cloud_src, point_cloud_tgt)  # shape: b x n_points_src x n_points_tgt
    dist_tgt_to_src = torch.cdist(point_cloud_tgt, point_cloud_src)  # shape: b x n_points_tgt x n_points_src

    # Calculate the minimum distance for each point in the source cloud.
    min_dist_src_to_tgt, _ = torch.min(dist_src_to_tgt, dim=2)  # shape: b x n_points_src
    min_dist_tgt_to_src, _ = torch.min(dist_tgt_to_src, dim=2)  # shape: b x n_points_tgt

    # Compute the Chamfer loss as the average of the minimum distances for both clouds.
    loss_chamfer = torch.mean(min_dist_src_to_tgt) + torch.mean(min_dist_tgt_to_src)
    return loss_chamfer

# Smoothness loss measures the smoothness of a mesh using Laplacian smoothing.
def smoothness_loss(mesh_src):
    smoothness_loss = mls(meshes=mesh_src, method="uniform")
    return smoothness_loss
