import torch
import matplotlib.pyplot as plt
plt.switch_backend('tkagg')
from mpl_toolkits.mplot3d import axes3d

def get_rays(height, width, focal_length, c2w ,device=None):
    i, j = torch.meshgrid(torch.arange(height,dtype=torch.float32).to(device),
                          torch.arange(width,dtype=torch.float32).to(device),indexing='ij')
    
    directions = torch.stack([(j - width * 0.5) / focal_length,
                            -(i - height * 0.5) / focal_length, 
                            -torch.ones_like(i)],dim=-1)

    # rays_d =  (c2w[:3,:3] @ directions[...,None]).squeeze(-1)
    rays_d = torch.einsum('...ij,...j->...i',c2w[:3,:3], directions)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return  rays_o, rays_d


def get_chunks(inputs, chunksize=2**15):
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_chunks(points, encoding_function, chunksize = 2**15):
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points

def prepare_viewdirs_chunks(  points, rays_d, encoding_function, chunksize = 2**15):
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def show_camera(origins, dirs):
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
    origins[..., 0].flatten().cpu(),
    origins[..., 1].flatten().cpu(),
    origins[..., 2].flatten().cpu(),
    dirs[..., 0].flatten().cpu(),
    dirs[..., 1].flatten().cpu(),
    dirs[..., 2].flatten().cpu(), normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    plt.gca().set_aspect('equal')
    plt.show()