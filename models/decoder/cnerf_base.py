import torch.nn as nn
import torch.nn.functional as F
import torch

from . import rendering


class CNeRF(nn.Module):
    ''' Generator Class.

    Args:
        device (pytorch device): pytorch device
        z_dim (int): dimension of latent code z
        decoder (nn.Module): decoder network
        n_ray_samples (int): number of samples per ray
        resolution_vol (int): resolution of volume-rendered image
        neural_renderer (nn.Module): neural renderer
        fov (float): field of view
    '''

    def __init__(self, device, z_dim=256, z_dim_bg=128, decoder=None,
                 neural_renderer=None,
                 n_ray_samples=64, resolution_vol=16,
                 fov=50, radius=2, depth_range=[0,4],
                 use_max_composition=False):
        super().__init__()
        self.device = device
        self.n_ray_samples = n_ray_samples
        self.resolution_vol = resolution_vol

        self.depth_range = depth_range

        self.fov = nn.Parameter((torch.ones(1)*fov).to(device))
        self.radius = nn.Parameter((torch.ones(1)*radius).to(device))
        self.z_dim = z_dim
        self.use_max_composition = use_max_composition

        self.camera_matrix = rendering.get_camera_mat(fov=self.fov)

        self.decoder = decoder.to(device)

        if neural_renderer is not None:
            self.neural_renderer = neural_renderer.to(device)
        else:
            self.neural_renderer = None

    def forward(self, latent_codes, camera_matrices,
                mode="training", 
                res=None, n_steps=None, it=0,
                importance_sample_depths=None,
                return_depth_map=False,
                return_world_map=False,
                enable_neural_renderer=False,
                return_view_params=False,
                return_depths_estimates=False,
                subsample_to=None,
                subsample_patch=None,
                pixel_errors=None,
                ray_termination=None):

        decoder_codes = latent_codes

        outputs = self.volume_render_image(
                decoder_codes, camera_matrices,
                mode=mode, res=res, n_steps=n_steps, it=it, 
                importance_sample_depths=importance_sample_depths,
                return_depth_map=return_depth_map,
                return_world_map=return_world_map,
                subsample_to=subsample_to,
                subsample_patch=subsample_patch,
                ray_termination=ray_termination,
                pixel_errors=pixel_errors)
        features, depth_map, world_map, pixel_loc, di, sigmas = outputs['feat_map'], outputs['depth_map'], outputs['world_map'], outputs['pixel_loc'], outputs['di'], outputs['sigmas']

        if enable_neural_renderer and self.neural_renderer is not None:
            features = self.neural_renderer(features, latent_codes)
        return_vals = [features]
        if return_depth_map:
            return_vals.append(depth_map)
        if return_world_map:
            return_vals.append(world_map)
        if return_depths_estimates:
            return_vals.append((di, sigmas))
        if subsample_to or subsample_patch:
            return_vals.append(pixel_loc)
        return return_vals


    def get_intrinsic_camera(self, batch_size=32):
        camera_mat = rendering.get_camera_mat(fov=self.fov).repeat(batch_size, 1, 1)
        return camera_mat

    def get_extrinsic_camera(self, camera_position, up=None, target=None, radius=None):
        return rendering.camera_pose_to_extrinsic(camera_position, up, target, radius)

    def add_noise_to_interval(self, di):
        di_mid = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low = torch.cat([di[..., :1], di_mid], dim=-1)
        noise = torch.rand_like(di_low)
        ti = di_low + (di_high - di_low) * noise
        return ti

    def importance_sample(self, sampled_coords, sampled_vals, n_points, n_steps, last_dist=1e10):
        sampled_depths_bsize, sampled_depths_npoints, sampled_depths_res = sampled_vals.shape
#        dists = sampled_coords[..., 1:] - sampled_coords[..., :-1]
#        dists = torch.cat([dists, torch.ones_like(
#            sampled_coords[..., :1]) * last_dist], dim=-1)
#        sampled_vals = 1.-torch.exp(-F.relu(sampled_vals)*dists)
        interleave_ratio = n_points // sampled_depths_npoints
        sampled_vals = sampled_vals.repeat_interleave(interleave_ratio, dim=1)
        sampled_coords = sampled_coords.repeat_interleave(interleave_ratio, dim=1)
        bin_edges = torch.cat([sampled_coords[...,0:1] * 0 + self.depth_range[0], sampled_coords, sampled_coords[...,0:1] * 0 + self.depth_range[1]], dim=-1)
        bin_weights = .5 * torch.cat([sampled_vals, sampled_vals[...,0:1] * 0], dim=-1) + torch.cat([sampled_vals[...,0:1] * 0, sampled_vals], dim=-1)
        bin_weights = bin_weights / bin_weights.sum(-1, keepdim=True)

        n_bins = bin_weights.size(-1)
        bin_weights = bin_weights.unsqueeze(-2)
        _, bin_choice = (bin_weights.cumsum(dim=-1)>torch.rand_like(bin_weights[...,:1].repeat(1,1,n_steps,1))).max(dim=-1)
        bin_start = torch.gather(bin_edges, -1, bin_choice)
        bin_end = torch.gather(bin_edges, -1, bin_choice+1)
        di = bin_start + torch.rand_like(bin_start) * (bin_end - bin_start)
        di, _ = torch.sort(di)
        return di



    def get_evaluation_points(self, pixels_world, camera_world, di,):

        batch_size = pixels_world.shape[0]
        n_steps = di.shape[-1]

        ray_world = pixels_world - camera_world

        p = camera_world.unsqueeze(-2).contiguous() + \
            di.unsqueeze(-1).contiguous() * \
            ray_world.unsqueeze(-2).contiguous()
        r = ray_world.unsqueeze(-2).repeat(1, 1, n_steps, 1)
        assert(p.shape == r.shape)
        p = p.reshape(batch_size, -1, 3)
        r = r.reshape(batch_size, -1, 3)
        return p, r



    def calc_volume_weights(self, z_vals, ray_vector, sigma, last_dist=1e1):
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(
            z_vals[..., :1]) * last_dist], dim=-1)

        dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)
        weights = alpha * \
            torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        return weights



    def volume_render_image(self, latent_codes, camera_matrices,
                            mode='training', res=None, n_steps=None,
                            it=0, importance_sample_depths=None,
                            return_depth_map=False,
                            return_world_map=False,
                            subsample_to=None,
                            subsample_patch=None,
                            ray_termination=None,
                            pixel_errors=None):
        feat_map = None
        acc_map = None
        depth_map = None
        world_map = None
        pixel_loc = None
        noise_weight = 1e0
        if res is None:
            res = self.resolution_vol
        if n_steps is None:
            n_steps = self.n_ray_samples
        device = self.device
        n_points = subsample_to if subsample_to is not None else res*res
        n_points = subsample_patch[0]*subsample_patch[1] if subsample_patch is not None else n_points
        depth_range = self.depth_range
        batch_size = latent_codes.shape[0]


        # Arange Pixels
        pixel_loc, pixels = rendering.arange_pixels((res, res), batch_size,
                               subsample_to=subsample_to, subsample_patch=subsample_patch, invert_y_axis=False, pixel_errors=pixel_errors)
        pixel_loc = pixel_loc.to(device)
        pixels = pixels.to(device)
        pixels[..., -1] *= -1.
        # Project to 3D world
        pixels_world = rendering.image_points_to_world(
            pixels, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
        camera_world = rendering.origin_to_world(
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])

#        camera_world = camera_world.index_select(pixel_loc)

        ray_vector = pixels_world - camera_world
        # batch_size x n_points x n_steps
        assert importance_sample_depths is None or ray_termination is None
        if importance_sample_depths is None and ray_termination is None:
            di = depth_range[0] + \
                torch.linspace(0., 1., steps=n_steps).reshape(1, 1, -1) * (
                    depth_range[1] - depth_range[0])
            di = di.repeat(batch_size, n_points, 1).to(device)
        elif importance_sample_depths is not None:
            sampled_coords, sampled_vals = importance_sample_depths
            sampled_depths_bsize, sampled_depths_npoints, sampled_depths_res = sampled_vals.shape
            assert batch_size == sampled_depths_bsize
            assert n_points / sampled_depths_npoints == n_points // sampled_depths_npoints
#            di = self.importance_sample(sampled_coords, sampled_vals, n_points=n_points, n_steps=n_steps)
            bins = .5 * (sampled_coords[:,:,1:] + sampled_coords[:,:,:-1]).view(-1, sampled_depths_res-1)
            weights = nn.functional.softmax(sampled_vals.view(-1, sampled_depths_res)[:,1:-1], dim=-1)
            di = sample_pdf(bins, weights, n_steps).detach().view(sampled_depths_bsize, sampled_depths_npoints, n_steps)
            di, _ = torch.sort(di, -1)  

        else:
            di = ray_termination.permute(0,3,2,1).reshape(batch_size, -1, n_steps)


        if mode == 'training': 
            di = self.add_noise_to_interval(di)

        p, r = self.get_evaluation_points(
            pixels_world, camera_world, di)

        feat, sigma = self.decoder(p, r, latent_codes)
        sigma = sigma.reshape(batch_size, n_points, n_steps)
        feat = feat.reshape(batch_size, n_points, n_steps, -1)

        if mode == 'training':
            # As done in NeRF, add noise during training
            sigma += noise_weight * torch.randn_like(sigma)


        # Get Volume Weights
        if n_steps != 1:
            weights = self.calc_volume_weights(di, ray_vector, sigma)
            feat_map = torch.sum(weights.unsqueeze(-1) * feat, dim=-2)
        else:
            weights = torch.ones_like(di)
            feat_map = feat.squeeze(-2)
        feat_map = feat_map + 1 - weights.sum(-1, keepdim=True)

        # Reformat output
        if subsample_to is None and subsample_patch is None:
            feat_map = feat_map.permute(0, 2, 1).reshape(
                batch_size, -1, res, res)  # B x feat x h x w
            feat_map = feat_map.permute(0, 1, 3, 2)  # new to flip x/y
        else:
            feat_map = feat_map.permute(0, 2, 1)  # new to flip x/y

        if return_depth_map:
            depth_map = torch.sum(weights * di, dim=-1, keepdim=True)

            # Reformat output
            if subsample_to is None and subsample_patch is None:
                depth_map = depth_map.permute(0, 2, 1).reshape(
                batch_size, -1, res, res)  # B x feat x h x w
                depth_map = depth_map.permute(0, 1, 3, 2)  # new to flip x/y
            else:
                depth_map = depth_map.permute(0, 2, 1)  # new to flip x/y

        if return_world_map:
            world_points = p.reshape(batch_size, n_points, n_steps, -1)
            world_map = torch.sum(weights.unsqueeze(-1) * world_points, dim=-2)

            # Reformat output
            if subsample_to is None and subsample_patch is None:
                world_map = world_map.permute(0, 2, 1).reshape(
                batch_size, -1, res, res)  # B x feat x h x w
                world_map = world_map.permute(0, 1, 3, 2)  # new to flip x/y
            else:
                world_map = world_map.permute(0, 2, 1)  # new to flip x/y
           
        return {'feat_map': feat_map,  'depth_map': depth_map, 'world_map': world_map, 'pixel_loc': pixel_loc, 'di': di, 'sigmas': sigma}



def reorder_tensor(t, scale=8, dim=-1):
    t_dims = t.shape
    out_dims = list(t.shape)
    out_dims[1] //= scale
    out_dims[dim] *= scale
    block_size = t.size(1)//scale
    blocks = [t[:,group*block_size:(group+1)*block_size] for group in range(scale)]
    t_out = torch.zeros(out_dims, device=t.device)
    t_out = t_out.transpose(0, dim)
    for i, b in enumerate(blocks):
        t_out[i::scale] = b.transpose(0, dim)
    t_out = t_out.transpose(0, dim)
    return t_out



def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf.detach(), u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples

