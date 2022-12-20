import torch
import torch.nn as nn

import torchvision

from .decoder.cnerf_base import CNeRF
from .decoder.cnerf_nn import NeRF_net



class ViewNeRF(nn.Module):
    """
    Base autoencoder architecture
    """

    def __init__(self, mod_conf, device=None, in_channels=3, out_channels=3):
        super(ViewNeRF, self).__init__()
        self.cont_size = mod_conf['cont_size']
        self.n_heads = mod_conf['n_heads']
        self.pose_net = EfficientEncoder(3 * self.n_heads + 3 + 3 + 3 + 1, detach_feats=False)
        self.code_net = EfficientEncoder(self.cont_size, detach_feats=False)

        render_conf = mod_conf['renderer']
        n_ray_samples=render_conf['n_ray_samples']
        resolution_vol=render_conf['resolution_vol']
        depth_range=render_conf['depth_range']
        radius=render_conf['radius']
        fov=render_conf['fov']

        dec_conf = mod_conf['decoder']
        hidden_size=dec_conf['hidden_size']
        use_viewdirs=dec_conf['use_viewdirs']
        initial_conditioning=dec_conf['initial_conditioning']
        downscale_by=dec_conf['downscale_by']
        n_freq_posenc=dec_conf['n_freq_posenc']
        n_freq_posenc_view=dec_conf['n_freq_posenc_view'] 
        skips=dec_conf['skips'] 
        n_blocks_shape=dec_conf['n_blocks_shape']
        n_blocks_view=dec_conf['n_blocks_view']
        final_sigmoid_activation=dec_conf['final_sigmoid_activation'] 
        rgb_out_dim=dec_conf['rgb_out_dim']

        nerf_net = NeRF_net(hidden_size=hidden_size, z_dim=self.cont_size,
            use_viewdirs=use_viewdirs, initial_conditioning=initial_conditioning,
            downscale_by=downscale_by,
            n_freq_posenc=n_freq_posenc, n_freq_posenc_view=n_freq_posenc_view, 
            skips=skips, n_blocks_shape=n_blocks_shape, n_blocks_view=n_blocks_view,
            final_sigmoid_activation=final_sigmoid_activation, rgb_out_dim=rgb_out_dim,)

        self.decoder = CNeRF(device=device, z_dim=self.cont_size, decoder=nerf_net,
            n_ray_samples=n_ray_samples, resolution_vol=resolution_vol, 
            depth_range=depth_range, radius=radius, fov=fov,
            )
       


    def forward(self, p_inp, c_inp, istrain=False, gt_pose=None):
        n_heads = self.n_heads
        bsize = p_inp.size(0)

        pose = self.pose_net(p_inp).view(bsize,-1)

        student, up, poses = pose[:,:n_heads], pose[:,n_heads:n_heads+3], pose[:,n_heads+3:]
        poses = poses.view(bsize,3,-1)
        norms = poses.norm(dim=-2)
        poses = poses/norms.unsqueeze(-2)

        latent_code = self.code_net(c_inp).view(bsize,-1)
        outputs = segmask = p_inp
        ""

        if istrain:
            outputs, segmask = self.decoder(latent_code, poses)
        else:
            _, inds = student.max(dim=1)
            pose = torch.stack([poses[k, :, head] for k, head in enumerate(inds)])
            outputs, segmask = self.decoder(latent_code, pose.unsqueeze(-1))
            poses = pose
        ""

        return outputs, segmask, student, poses, latent_code

    def name(self):
        return "ViewNeRF"

class EfficientEncoder(nn.Module):

    def __init__(self, out_size, reduce=True, replace_head=True, detach_feats=True):
        super(EfficientEncoder, self).__init__()
        self.detach_feats = detach_feats
        self.net = torchvision.models.efficientnet_b0(pretrained=True)
        if not reduce:
            self.net.avgpool = nn.Identity()
        if replace_head:
            self.net.classifier = nn.Linear(1280, out_size)
        else:
            self.net.classifier = nn.Identity()
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])

    def forward(self, x):
        x = self.normalize(x)
        feats = self.net.features(x)
        if self.detach_feats:
            feats = feats.detach()
        avg = self.net.avgpool(feats).flatten(1)
        out = self.net.classifier(avg)
        return out
