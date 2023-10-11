import torch
import numpy as np

# Blender is right hand system
def dataset_loader():
    data = np.load('../ganyu_150.npz')
    # data = np.load('../tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses'] # camera to world
    focal = data['focal']
    return images, poses, focal