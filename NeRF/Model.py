import torch
from torch import nn
import torch.nn.functional as F
from utils import sample_stratified, sample_hierarchical
from utils.Camera import prepare_chunks, prepare_viewdirs_chunks
from rendering import raw2outputs


class NeRF_MLP(nn.Module):
    def __init__(self, d_input=3, n_layers=8, d_hidden=256, skip=(4,), d_viewdirs = None):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.skip = skip
        self.act = F.relu
        self.d_viewdirs = d_viewdirs
        
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, self.d_hidden)] + 
            [nn.Linear(self.d_hidden, self.d_hidden) if i not in skip 
                else nn.Linear(self.d_hidden + self.d_input, self.d_hidden)
                for i in range(n_layers-1)]
        )
        
        if self.d_viewdirs is not None:
            self.alpha_out = nn.Linear(self.d_hidden, 1)
            self.rgb_hidden = nn.Linear(self.d_hidden, self.d_hidden)
            self.branch = nn.Linear(self.d_hidden + self.d_viewdirs, self.d_hidden//2)
            self.output = nn.Linear(d_hidden//2, 3)
        else:
            self.output = nn.Linear(d_hidden, 4)
    

    def forward(self, x, viewdirs=None):
        # assert self.d_viewdirs is None and viewdirs is not None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x,x_input], dim=-1)
        
        if self.d_viewdirs is not None:      
            alpha = self.alpha_out(x)
            x = self.rgb_hidden(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.branch(x)
            x = self.act(x)
            x = self.output(x)
            x = torch.concat([x, alpha], dim=-1)
        else:
            x = self.output(x)
        return x


def nerf_forward(rays_o, rays_d, near, far, encoding_fn, coarse_model,  kwargs_sample_stratified = None, 
                 n_samples_hierarchical = 0, kwargs_sample_hierarchical = None,  fine_model = None, 
                 viewdirs_encoding_fn = None, chunksize: int = 2**15,device=None):

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}
  
    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(rays_o, rays_d, near, far, **kwargs_sample_stratified, device=device)

    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d, viewdirs_encoding_fn,chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

  # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
  # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {
        'z_vals_stratified': z_vals
    }

  # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map
        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical( rays_o, rays_d, z_vals, weights, n_samples_hierarchical,**kwargs_sample_hierarchical,device=device)

    # Prepare inputs as before.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,   viewdirs_encoding_fn, chunksize=chunksize)
    else:
        batches_viewdirs = [None] * len(batches)

    # Forward pass new samples through fine model.
    fine_model = fine_model if fine_model is not None else coarse_model
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)
    
    # Store outputs.
    outputs['z_vals_hierarchical'] = z_hierarch
    outputs['rgb_map_0'] = rgb_map_0
    outputs['depth_map_0'] = depth_map_0
    outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs