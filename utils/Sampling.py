import torch

def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True, inverse_depth=False, device=None):
    t_vals = torch.linspace(0., 1., n_samples, device=device)
    if not inverse_depth:
        # Sample linearly between near and far
        z_vals = near * (1.-t_vals) + far * (t_vals)  # near + (far - near) * t_vals
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[1:] + z_vals[:-1])  # 中点
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=device)
        z_vals = lower + (upper - lower) * t_rand

    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])
    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


def sample_pdf(bins,  weights, n_samples, perturb = False, device=None):
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]
    # convert PDF to cumulative distribution function(CDF) 对PDF进行积分
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    indexs = torch.searchsorted(cdf, u, right=True)
    
    # Clamp indices that are out of bounds.
    below = torch.clamp(indexs - 1, min=0)
    above = torch.clamp(indexs, max=cdf.shape[-1] - 1)
    indexs_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]
    
    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=indexs_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=indexs_g)

    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=device)
        u = u.expand(list(cdf.shape[:]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:]) + [n_samples], device=device) # [n_rays, n_samples]
        
    u = u.contiguous()
    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples # [n_rays, n_samples] 


def sample_hierarchical(  rays_o,  rays_d,  z_vals,  weights,  n_samples,  perturbFalse):
    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples
    


def show_samples(rays_o, rays_d, near, far, n_samples, perturb=False, inverse_depth=False, device=None):
    _, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples,perturb=True, inverse_depth=inverse_depth)
    y_vals = torch.zeros_like(z_vals)
    _, z_vals_unperturbed = sample_stratified(rays_o, rays_d, near, far, n_samples,  perturb=False, inverse_depth=inverse_depth)
    plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), 'b-o')
    plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), 'r-o')
    plt.ylim([-1, 2])
    plt.title('Stratified Sampling (blue) with Perturbation (red)')
    ax = plt.gca()
    ax.axes.yaxis.set_visible(False)
    plt.grid(True)
    plt.show()