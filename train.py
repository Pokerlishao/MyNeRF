import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import trange
from typing import Optional, Tuple, List, Union, Callable


from datasets import dataset_loader
from utils import get_rays, show_camera, sample_stratified, show_samples
from utils import PositionalEncoder
from NeRF import NeRF_MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images, poses, focal = dataset_loader()

print(f'Images shape: {images.shape}')
print(f'Poses shape: {poses.shape}')
print(f'Focal length: {focal}')
near, far = 2., 6.
n_training = 30
testimg_idx = 33
images = torch.from_numpy(images).to(device)
poses = torch.from_numpy(poses).to(device)
focal = torch.from_numpy(focal).to(device)
height, width = images.shape[1:3]
testimg = images[testimg_idx]
testpose = poses[testimg_idx]




with torch.no_grad():
    ray_origin, ray_direction = get_rays(height, width, focal, testpose,device)
    # Draw stratified samples from example
    rays_o = ray_origin.view([-1, 3])
    rays_d = ray_direction.view([-1, 3])


n_samples = 8
perturb = True
inverse_depth = False
with torch.no_grad():
    pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples,
                                  perturb=perturb, inverse_depth=inverse_depth,device=device)
    
    # show_samples(rays_o, rays_d, near, far, n_samples,
    #                               perturb=perturb, inverse_depth=inverse_depth,device=device)


encoder = PositionalEncoder(3, 10)
viewdirs_encoder = PositionalEncoder(3, 4)
# Grab flattened points and view directions
pts_flattened = pts.reshape(-1, 3)
viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

# Encode inputs
encoded_points = encoder(pts_flattened)
encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

# mynerf = NeRF_MLP(3,8,256,(4,),2)
# y = mynerf(torch.tensor([1.,2.,3.]), torch.tensor([1.,2.]))
# print(y)


# def main():
#     dirs = torch.stack([(pose[:3, :3] @ torch.tensor([0., 0., -1.]).to(device)) for pose in poses])
#     origins = poses[:, :3, -1]
#     show_camera(origins, dirs)
    
# if __name__ == "__main__":
#     main()




# with torch.no_grad():
#     ray_origin, ray_direction = get_rays(height, width, focal, testpose, device)
    
# print(ray_direction)

    