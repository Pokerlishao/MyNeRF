import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import trange
from typing import Optional, Tuple, List, Union, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange


from datasets import dataset_loader
from utils import get_rays, show_camera, sample_stratified, show_samples
from utils import PositionalEncoder
from NeRF import NeRF_MLP, nerf_forward

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

oimages, poses, focal = dataset_loader()

print(f'Images shape: {oimages.shape}')
print(f'Poses shape: {poses.shape}')
print(f'Focal length: {focal}')
near, far = 2., 6.
n_training = 30
testimg_idx = 33

images = torch.from_numpy(oimages[:n_training]).to(device)
poses = torch.from_numpy(poses).to(device)
focal = torch.from_numpy(focal).to(device)

testimg = torch.from_numpy(oimages[testimg_idx]).to(device)
testpose = poses[testimg_idx]
height, width = images.shape[1:3]



# Encoders
d_input = 3           # Number of input dimensions
n_freqs = 10          # Number of encoding functions for samples
log_space = True      # If set, frequencies scale in log space
use_viewdirs = True   # If set, use view direction as input
n_freqs_views = 4     # Number of encoding functions for views

# Stratified sampling
n_samples = 64         # Number of spatial samples per ray
perturb = True         # If set, applies noise to sample positions
inverse_depth = False  # If set, samples points linearly in inverse depth

# Model
d_hidden = 128          # Dimensions of linear layer filters
n_layers = 2            # Number of layers in network bottleneck
skip = []               # Layers at which to apply input residual
use_fine_model = True   # If set, creates a fine model
d_filter_fine = 128     # Dimensions of linear layer filters of fine network
n_layers_fine = 6       # Number of layers in fine network bottleneck

# Hierarchical sampling
n_samples_hierarchical = 64   # Number of samples per ray
perturb_hierarchical = False  # If set, applies noise to sample positions

# Optimizer
lr = 5e-4  # Learning rate

# Training
n_iters = 10000
batch_size = 2**14          # Number of rays per gradient step (power of 2)
one_image_per_step = False   # One image per gradient step (disables batching)
chunksize = 2**14           # Modify as needed to fit in GPU memory
center_crop = True          # Crop the center of image (one_image_per_)
center_crop_iters = 50      # Stop cropping center after this many epochs
display_rate = 50         # Display test output every X epochs

# We bundle the kwargs for various functions to pass all at once.
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}
kwargs_sample_hierarchical = {
    'perturb': perturb
}


def init_models():
    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    encode = lambda x: encoder(x)

    # View direction encoders
    if use_viewdirs:
        encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,log_space=log_space)
        encode_viewdirs = lambda x: encoder_viewdirs(x)
        d_viewdirs = encoder_viewdirs.d_output
    else:
        encode_viewdirs = None
        d_viewdirs = None

    # Models
    model = NeRF_MLP(encoder.d_output, n_layers=n_layers, d_hidden=d_hidden, skip=skip, d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())
    if use_fine_model:
        fine_model = NeRF_MLP(encoder.d_output, n_layers=n_layers, d_hidden=d_hidden, skip=skip, d_viewdirs=d_viewdirs)
        fine_model.to(device)
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)
    return model, fine_model, encode, encode_viewdirs, optimizer

def crop_center(img, frac= 0.5):
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]

def train():
    model, fine_model, encode, encode_viewdirs, optimizer = init_models()
    if not one_image_per_step:
        height, width = images.shape[1:3]
        all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p, device=device), 0)
                            for p in poses[:n_training]], 0)
        rays_rgb = torch.cat([all_rays, images[:, None]], 1)
        rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = rays_rgb.reshape([-1, 3, 3])
        rays_rgb = rays_rgb.type(torch.float32)
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
        i_batch = 0
        
    train_psnrs = []
    val_psnrs = []
    iternums = []

    for i in trange(n_iters):
        model.train()
        if one_image_per_step:
            # Randomly pick an image as the target.
            target_img_idx = np.random.randint(images.shape[0])
            target_img = images[target_img_idx].to(device)
            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]
            target_pose = poses[target_img_idx].to(device)
            rays_o, rays_d = get_rays(height, width, focal, target_pose)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])

        else:
            # Random over all images.
            batch = rays_rgb[i_batch:i_batch + batch_size]
            batch = torch.transpose(batch, 0, 1)
            rays_o, rays_d, target_img = batch
            height, width = target_img.shape[:2]
            i_batch += batch_size
            # Shuffle after one epoch
            if i_batch >= rays_rgb.shape[0]:
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
                i_batch = 0
        target_img = target_img.reshape([-1, 3])
        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = nerf_forward(rays_o, rays_d, near, far, encode, model, kwargs_sample_stratified=kwargs_sample_stratified,
                               n_samples_hierarchical=n_samples_hierarchical, kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                               fine_model=fine_model, viewdirs_encoding_fn=encode_viewdirs, chunksize=chunksize, device=device)
        

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        rgb_predicted = outputs['rgb_map']
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())
        
            # Evaluate testimg at given display rate.
        if i % display_rate == 0:
            model.eval()
            height, width = testimg.shape[:2]
            rays_o, rays_d = get_rays(height, width, focal, testpose,device=device)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            outputs = nerf_forward(rays_o, rays_d,
                                    near, far, encode, model,
                                    kwargs_sample_stratified=kwargs_sample_stratified,
                                    n_samples_hierarchical=n_samples_hierarchical,
                                    kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                    fine_model=fine_model,
                                    viewdirs_encoding_fn=encode_viewdirs,
                                    chunksize=chunksize,device=device)

            rgb_predicted = outputs['rgb_map']
            loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
            val_psnr = -10. * torch.log10(loss)
            val_psnrs.append(val_psnr.item())
            iternums.append(i)
            plt.imsave(f'{i}.jpg',rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
            print("Loss:", loss.item())
            print(f'Loss: {loss.item()}, PSNR: {val_psnr}')

    return True, train_psnrs, val_psnrs


def main():
    success, train_psnrs, val_psnrs = train()
    if success:
        print(f'Done! Train PSNR: {train_psnrs}, Val PSNR: {val_psnrs}')
    
if __name__ == "__main__":
    main()
