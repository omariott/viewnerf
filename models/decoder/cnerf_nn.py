import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi


class NeRF_net(nn.Module):
    ''' NeRF_net class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_view (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''

    def __init__(self, hidden_size=128, n_blocks_shape=8, n_blocks_texture=0, n_blocks_view=1,
                 skips=[4], use_viewdirs=True, n_freq_posenc=10,
                 n_freq_posenc_view=4,
                 z_dim=64, rgb_out_dim=128, final_sigmoid_activation=False,
                 downscale_by=1., positional_encoding="normal",
                 gauss_dim_pos=10, gauss_dim_view=4, gauss_std=4.,
                 initial_conditioning=True, skip_connections=False,):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_view = n_freq_posenc_view
        self.skips = skips
        self.downscale_by = downscale_by
        self.z_dim = z_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks_texture = n_blocks_texture
        self.n_blocks_shape = n_blocks_shape
        self.n_blocks_view = n_blocks_view
        self.initial_conditioning = initial_conditioning
        self.skip_connections = skip_connections
        
        assert(positional_encoding in ('normal', 'gauss', 'none'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        elif positional_encoding == 'normal':
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_view * 2
        else:
            dim_embed = 3
            dim_embed_view = 3

        # Density Prediction Layers
        self.fc_in = nn.Linear(dim_embed, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
            )
            for i in range(n_blocks_shape - 1)
        ])
        n_skips = sum([i in skips for i in range(n_blocks_texture + n_blocks_shape - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList(
                [nn.Linear(z_dim, hidden_size) for i in range(n_skips)]
            )
            self.fc_p_skips = nn.ModuleList([
                nn.Linear(dim_embed, hidden_size) for i in range(n_skips)]
            )
        self.sigma_out = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.sigma_out.bias[0] = 0
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)

        # Feature Prediction Layers
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        if use_viewdirs and n_blocks_view > 1:
            self.blocks_view = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size)
                 for i in range(n_blocks_view - 1)])

    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        elif self.positional_encoding == 'normal':
            L = self.n_freq_posenc_view if views else self.n_freq_posenc
#            norm = p.norm(dim=-1, keepdim=True)
#            p_padded = torch.cat([p,torch.ones_like(p[...,0:1])], dim=-1)
#            p_normed = p_padded * norm.pow(-1)
#            p = torch.where(norm < 1, p_padded, p_normed)
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        else:
            p_transformed = p
        return p_transformed

 

    def forward(self, p_in, ray_d, code, **kwargs):
        batch_size, n_samples = p_in.shape[0], p_in.shape[1]
        a = F.relu 
        if self.z_dim > 0:
            if code is None:
                code = torch.zeros(batch_size, self.z_dim).to(p_in.device) 
        p = self.transform_points(p_in)
        net = self.fc_in(p)
        if self.initial_conditioning:
            net = net + self.fc_z(code).unsqueeze(1)
        net = a(net)


        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            if self.skip_connections:
                net = net + layer(net)
            else:
                net = layer(net)
            if idx in self.skips:
                pose_mask = self.fc_p_skips[skip_idx](p)
                net = net + self.fc_z_skips[skip_idx](code).unsqueeze(1) + pose_mask
                skip_idx += 1
            net = a(net)
        sigma_out = self.sigma_out(net).squeeze(-1)

        if self.use_viewdirs and ray_d is not None:
#            ray_di = ray_d[indices]
            ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            ray_d = self.transform_points(ray_d, views=True)
            net = net + self.fc_view(ray_d)
            net = a(net)
            if self.n_blocks_view > 1:
                for layer in self.blocks_view:
                    net = a(layer(net))
        feat_out = self.feat_out(net)

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out
