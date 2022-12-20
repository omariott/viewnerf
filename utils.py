import numpy as np
import os
import torch
import torch.nn as nn
import imageio


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, keep_all=False):
        self.reset()
        self.data = None
        if keep_all:
            self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.data is not None:
            self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MedianMeter(object):
    """Computes and stores the Median and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.data = []

    def update(self, vals):
        self.data.append(vals)
        self.val = torch.cat(self.data).median().item()



class Code_logger(object):
    """logs aribitrary tensors as .npy"""
    def __init__(self, path="logs"):
        self.path = path
        self.codes = dict()

    def log_code(self, key, code):
        if key in self.codes:
            self.codes[key].append(code.detach().cpu().numpy())
        else:
            self.codes[key] = [code.detach().cpu().numpy()]

    def print_code(self, name):
        if self.codes:
            full_code = np.stack([np.concatenate(vals) for vals in self.codes.values()], axis=-1)
            np.save(os.path.join(self.path, name+"_codelog"), full_code)


    def reset_codes(self):
        self.codes = dict()


class Model_logger(object):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    def dump(self, name):
        path = os.path.join(self.path, name)+".pth"
        torch.save(self.model.state_dict(), path)


def classification_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def rel_angle(output, target):
    n_dim = len(target.shape)
    in_dim = target.size(1)
    if n_dim == 3:
        angle = rotmat_angle(output, target)
    elif in_dim==4:
        angle = angle_quat(output, target)
    elif in_dim==3:
        angle = angle_vec(output, target)
    else:
        print("Error, target should contain 3 or 4 channels but got {}".format(in_dim))
        exit()
    return angle

def rotmat_angle(A,B):
    rots = A.transpose(-1,-2).bmm(B)
    tr = torch.stack([r.trace() for r in rots])
    th = ((tr - 1) * .5).acos()
    return th/np.pi*180

def angle_quat(output, target):
    angle = (2*(output*target).sum(-1).pow(2)-1).acos()
    return angle.squeeze()/np.pi*180


def angle_vec(output, target):
    angle = (output*target).sum(-1).acos().float()
    return angle.squeeze()/np.pi*180


def angle_accuracy(output, target, thresh=10):
    bsize = target.size(0)
    angle = rel_angle(output, target)
    correct = angle < thresh
    res = correct.float().sum()/bsize
    return res


def theta2sphere(theta):
    azi, elev = theta[:,0], theta[:,1]
    x = elev.cos()*azi.cos()
    y = elev.cos()*azi.sin()
    z = elev.sin()
    code = torch.stack([x, y, z], dim=1)
    return code

def sphere2theta(points):
    x, y, z = points[:,0], points[:,1], points[:,2]
    az = torch.atan2(x,y)
    elev = torch.asin(z)
    code = torch.stack([az, elev], dim=1)
    return(code)

def az2quat(azimuth):
    zeros = torch.zeros_like(azimuth)
    quat = torch.stack([(azimuth/2).cos(), zeros, zeros, (azimuth/2).sin()], dim = -1)
    return quat

def el2quat(elevation):
    zeros = torch.zeros_like(elevation)
    quat = torch.stack([(elevation/2).cos(), zeros, (elevation/2).sin(), zeros], dim = -1)
    return quat

def angle2quat(angles):
    azimuth, elevation = angles.unbind(dim=1)
    return quatprod(az2quat(azimuth), el2quat(elevation))

def quatprod(q1, q2):
#    print("q1 :",q1.shape)
#    print("q2 :",q2.shape)
    w1, x1, y1, z1 = q1.unbind(dim=1)
    w2, x2, y2, z2 = q2.unbind(dim=1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    quat = torch.stack([w,x,y,z], dim=1)
#    print("prod :",quat.shape)
    return quat

def conj(q):
    conjugator = -1*torch.ones_like(q)
    conjugator[:,0] = 1
    return q*conjugator

def quatrot(v, q):
    while(len(q.shape)<len(v.shape)):
        q = q.unsqueeze(-1)
    zeros = torch.zeros_like(v[:,0:1])
    extended_v = torch.cat([zeros,v], dim=1)
    return quatprod(quatprod(q, extended_v), conj(q))[:,1:]

def quat2rotmat(q):
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    rotmat = torch.stack(
        [
            torch.stack([1-2*y.pow(2)-2*z.pow(2), 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
            torch.stack([2*x*y + 2*z*w, 1-2*x.pow(2)-2*z.pow(2), 2*y*z - 2*x*w], dim=-1),
            torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1-2*x.pow(2)-2*y.pow(2)], dim=-1),
        ]
    , dim=-1)
    return rotmat


def angles2rotmat(az, el, ti):
    zeros = torch.zeros_like(az)
    ones = torch.ones_like(az)
    Mx = torch.stack([
        torch.stack([az.cos(), az.sin(), zeros], dim=1),
        torch.stack([-az.sin(), az.cos(), zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1),
    ], dim=1)
    My = torch.stack([
        torch.stack([el.cos(), zeros, el.sin()], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-el.sin(), zeros, el.cos()], dim=1),
    ], dim=1)
    Mz = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, ti.cos(), ti.sin()], dim=1),
        torch.stack([zeros, -ti.sin(), ti.cos()], dim=1),
    ], dim=1)
    return Mz.bmm(My.bmm(Mx))

def rotationMatrixToEulerAngles(R) :
    sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
    singular = sy < 1e-6
    x1 = torch.atan2(R[:,2,1] , R[:,2,2])
    y1 = torch.atan2(-R[:,2,0], sy)
    z1 = torch.atan2(R[:,1,0], R[:,0,0])
    x2 = torch.atan2(-R[:,1,2], R[:,1,1])
    y2 = torch.atan2(-R[:,2,0], sy)
    z2 = torch.zeros_like(x2)
    x = torch.where(singular, x2, x1)
    y = torch.where(singular, y2, y1)
    z = torch.where(singular, z2, z1)
    return x, y, z



def extract_pixels(images, pixel_locations):
    bsize, channels, hdim, wdim = images.shape
    _, samples, _ = pixel_locations.shape
    row, col = pixel_locations.unbind(-1)
    return torch.stack([images[b,:,col[b],row[b]] for b in range(bsize)])


def sample_compose(images, pixels, pixel_locations):
    bsize, channels, hdim, wdim = images.shape
    _, samples, _ = pixel_locations.shape
    row, col = pixel_locations.unbind(-1)
    for b in range(bsize):
        images[b, :, col[b], row[b]] = pixels[b]
    return images

def save_reconstruct(name, images, diff_inds=(0,-1), path="plots", diff=True):
    ims = []
    h,w = images[0].size(-2), images[0].size(-1)
    for i, image in enumerate(images):
        image = nn.functional.interpolate(image.unsqueeze(0), (h,w)).squeeze(0)
        pic = image.detach().cpu().numpy()
        ims.append(pic)
    if diff:
        diff = (images[diff_inds[0]]-images[diff_inds[1]]).detach().pow(2).cpu().numpy()
        ims.append(diff/diff.max())
    out = np.concatenate(ims, axis=2).transpose((1,2,0))
    out = (out*255).astype('uint8')
    imageio.imsave(os.path.join(path, name+"_reconstruct.png"), out)

def save_gif(batch, name, path):
    seq = batch.unbind(0)
    images = []
    for i, image in enumerate(seq):
        im = image.detach().cpu().numpy().transpose((1,2,0))
        im = (im * 255).round().astype(np.uint8)
        images.append(im)
    imageio.mimsave(os.path.join(path, name+".gif"), images)

def create_views(data, model, device, name, view_count=64, path='plots'):

    torch.autograd.set_grad_enabled(False)

    model.eval()

    for i, inp in enumerate(data):
        bsize = 1
        c_inp = inp['cont_im']
        c_inp = c_inp[:bsize]

        c_inp = c_inp.float().to(device)

        c_inp = nn.functional.interpolate(c_inp, (128, 128))
        outputs = []
        depths = []
        maxid = view_count//bsize
        ratio = bsize/view_count
        for batch_id in range(maxid):

            gen = model.decoder
            elevation = torch.ones(bsize, device=c_inp.device) * 5
            azimuth = torch.linspace(0, 1, bsize, device=c_inp.device) * (ratio - 1 / view_count) + ratio * batch_id
            azimuth = azimuth * 360

            c_pose = torch.stack([azimuth, elevation], dim=-1)
            c_pose = theta2sphere(c_pose.float()/180*np.pi)
            codes = model.code_net(c_inp)
            intrinsic = gen.get_intrinsic_camera(batch_size=bsize)
            extrinsic = gen.get_extrinsic_camera(c_pose, radius=gen.radius)
            camera_matrices = intrinsic, extrinsic
            
            pred, depth_map = gen(res=128, n_steps=128, latent_codes=codes, camera_matrices=camera_matrices, mode='val', return_depth_map=True,)

            outputs.append(pred)
            depths.append(depth_map)
        output = torch.cat(outputs, dim=0)
        depth_map = torch.cat(depths, dim=0)
        discrepancy = (1 + depth_map).pow(-1)
        save_gif(output, name, path)
        save_gif(discrepancy, name+"_depth", path)
        break

def pts2rot(kpts_a, kpts_b, noise=0.0):
    bsize = kpts_a.size(0)
    n_dim = kpts_a.size(-1)
    kpts_a = kpts_a + torch.randn_like(kpts_a)*noise
    kpts_b = kpts_b + torch.randn_like(kpts_b)*noise
    prod = kpts_a.transpose(-1,-2).bmm(kpts_b)
    u, _, vT = torch.svd(prod)
    det = torch.det(u.bmm(vT.transpose(-1,-2)))
    corrector = torch.eye(n_dim, device=vT.device).repeat(bsize,1,1)
    corrector[:,-1,-1] = det
    vT = torch.bmm(corrector.detach(), vT)
    rotmat = torch.bmm(u, vT.transpose(-1,-2))
    return rotmat


def align_preds(data, model, device, samples=100):
    torch.autograd.set_grad_enabled(False)
    model.eval()
    sample_count = 0
    preds = []
    gt = []
    for i, inp in enumerate(data):
        pose_inp, azimuth, elevation, idx = inp['pose_im'], inp['azimuth'], inp['elevation'], inp['idx']
        bsize = pose_inp.size(0)
        sample_count += bsize
        pose_inp = pose_inp.float().to(device)
        azimuth = azimuth.to(device)
        elevation = elevation.to(device)
        idx = idx.to(device)
        pose_inp = nn.functional.interpolate(pose_inp, (128, 128))
        gt_angles = torch.stack([azimuth, elevation], dim=1)
        gt_pose = theta2sphere(gt_angles.float()/180*np.pi)
        pred_pose = model.pose_net(pose_inp).view(bsize, -1)[:,0:3]
        pred_pose = pred_pose/pred_pose.norm(dim=-1, keepdim=True)
        preds.append(pred_pose)
        gt.append(gt_pose)
        if sample_count>samples:
            break
    preds = torch.cat(preds, dim=0)
    gt = torch.cat(gt, dim=0)
    samples_effective = min(samples, gt.size(0))
    preds, gt = preds[:samples_effective], gt[:samples_effective]
    rotmat =  pts2rot(gt.unsqueeze(0), preds.unsqueeze(0))
    return rotmat
