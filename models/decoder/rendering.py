import numpy as np
import torch

def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / torch.tan(0.5 * fov * np.pi/180.)
    mat = torch.eye(4, device=fov.device)
    mat[0,0] = mat[1,1] = focal
    mat = mat.reshape(1, 4, 4)

    if invert:
        mat = torch.inverse(mat)
    return mat


def camera_pose_to_extrinsic(camera_position, up=None, target=None, radius=None):
    if radius is not None:
        camera_position = camera_position / camera_position.norm(dim=-1, keepdim=True) * radius
    rotmat = lookat_camera_rotation(camera_position, up, target)
    padded_mat = torch.cat([rotmat, torch.zeros_like(rotmat[:,0:1,:])], dim=1) # bsize, 4, 3
    padded_trans = torch.cat([camera_position, torch.ones_like(camera_position[:,0:1])], dim=1).unsqueeze(-1) # bsize, 4, 1
    extrinsic = torch.cat([padded_mat, padded_trans], dim=-1)
    return extrinsic
  

def lookat_camera_rotation(camera_position, up=None, target=None,):
    """
        Recover camera rotation from position and looking point
    """
    if target is None:
        target = torch.zeros_like(camera_position)
    if up is None:    
        up = torch.zeros_like(camera_position)
        up[:,-1] += 1
    z = camera_position-target
    z = z/z.norm(dim=-1, keepdim=True)
    x = torch.cross(up,z, dim=-1)
    x = x/x.norm(dim=-1, keepdim=True)
    y = torch.cross(z,x, dim=-1)
    rotmat = torch.stack((x,y,z), dim=-1)
    return rotmat

def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None, subsample_patch=None, invert_y_axis=False,
                  pixel_errors=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    assert not (subsample_to and subsample_patch)
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h), indexing='ij')
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idxs = np.array([np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False, p=pixel_errors) for k in range(batch_size)])

#        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
#                               replace=False, p=pixel_errors)
#        idx_x = torch.rand(size=(subsample_to,))
#        idx_y = torch.rand(size=(subsample_to,))
#        idx = (idx_x * h).long() + w * (idx_y * w).long()

        pixel_scaled = torch.stack([pixel_scaled[k, idxs[k]] for k in range(batch_size)])
        pixel_locations = torch.stack([pixel_locations[k, idxs[k]] for k in range(batch_size)])
#        pixel_locatoins = torch.stack([idx_x, idx_y]).unsqueeze(-1).unsqueeze(-1)

    # draw random patch is subsample_patch is not None
    if (subsample_patch is not None and len(subsample_patch) == 2 and
            subsample_patch[0]*subsample_patch[1] < n_points):
        hp,wp = subsample_patch
        hstart, wstart = np.random.randint(0, h-hp), np.random.randint(0, w-wp)
        indstart = hstart * w + wstart
        idx = np.concatenate([np.arange(s, s+wp) for s in range(indstart, indstart + w * hp, w)])
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    if invert_y_axis:
        assert(image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled




def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat=None,
                       invert=True, use_absolute_depth=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(
            camera_mat.shape[0], 1, 1).to(camera_mat.device)
    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    return p_world



def image_points_to_world(image_points, camera_mat, world_mat, scale_mat=None,
                          invert=False, negative_depth=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    if negative_depth:
        d_image *= -1.
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


def origin_to_world(n_points, camera_mat, world_mat, scale_mat=None,
                    invert=False):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(
            0).repeat(batch_size, 1, 1).to(device)

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world
