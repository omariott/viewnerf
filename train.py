import argparse
import os
import shutil
import time
import yaml

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import numpy as np

from datasets import get_dataset
import models

from utils import AverageMeter, MedianMeter, Model_logger, Code_logger, theta2sphere, align_preds,angle_accuracy, rel_angle, save_reconstruct, create_views


def get_loader(dataset, batch_size):
    pin_memory = device.type == 'cuda'
    train_dataset, val_dataset, test_dataset = dataset

    data_len = len(train_dataset)
    train_dataset = torch.utils.data.Subset(train_dataset, range(0,data_len))
    replications = max(1, 32000//len(train_dataset))
    train_dataset = torch.utils.data.ConcatDataset([train_dataset]*replications)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, drop_last=False,
                        pin_memory=pin_memory, num_workers=10)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, drop_last=False,
                        pin_memory=pin_memory, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size, shuffle=False, drop_last=False,
                        pin_memory=pin_memory, num_workers=10)

    return train_loader, val_loader, test_loader


def epoch(e, data, model, criterion, optimizer=None, logger=None, split=None, align_mat=None, cfg=None):

    if not split in ["train", "val", "test"]:
        print("Invalid argument split : must be train, val or test, but got ", split)
        raise SystemExit

    istrain = (split=="train")
    torch.autograd.set_grad_enabled(istrain)

    model = model.train() if istrain else model.eval()

    avg_loss_rec = AverageMeter()
    avg_batch_time = AverageMeter()

    avg_acc = AverageMeter()
    med_err = MedianMeter()



    for i, inp in enumerate(data):

        tic = time.time()
        p_inp, c_inp, target, azimuth, elevation, pose_mat = inp['pose_im'], inp['cont_im'], inp['tar_im'], inp['azimuth'], inp['elevation'], inp['pose_mat']


        tilt = torch.zeros_like(elevation)
        bsize = p_inp.size(0)
        xsize = p_inp.size(-1)
        ysize = p_inp.size(-2)

        p_inp = p_inp.float().to(device)
        c_inp = c_inp.float().to(device)
        target = target.float().to(device)
        azimuth = azimuth.float().to(device)
        elevation = elevation.float().to(device)
        tilt = tilt.float().to(device)
        pose_mat = pose_mat.float().to(device)

        loss = 0
        in_res = cfg['training']['in_res']
        out_res = cfg['training']['out_res']
        p_inp = nn.functional.interpolate(p_inp, (in_res, in_res))
        c_inp = nn.functional.interpolate(c_inp, (in_res, in_res))
        target = nn.functional.interpolate(target, (out_res, out_res))


        pred_pose = model.pose_net(p_inp).view(bsize, -1)

        student, at, up, teacher, cam_dist = pred_pose[:,0:3], pred_pose[:,3:6], pred_pose[:,6:9], pred_pose[:,9:9+3*model.n_heads], pred_pose[:,9+3*model.n_heads:]
        if not cfg['training']['exact_sphere']:
            up_target = torch.zeros_like(up)
            up_target[:,-1] += 1

            if e<10:
                loss += 1e0 * nn.functional.mse_loss(up, up_target)
                loss += 1e0 * nn.functional.mse_loss(at, torch.zeros_like(at))

        student = student/student.norm(dim=-1, keepdim=True)
        codes = model.code_net(c_inp).squeeze()

        gen = model.decoder

        mode = 'training' if istrain else 'val'


        if istrain and model.n_heads>1:
            dists = []
            teacher = teacher.view(bsize, -1, 3).transpose(0,1)
            teacher = teacher/teacher.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                for pose in teacher:
                    intrinsic = gen.get_intrinsic_camera(batch_size=bsize)
                    if not cfg['training']['exact_sphere']:
                        radius = cam_dist.sigmoid().add(1e-5).pow(-1).add(-1).mul(gen.radius)
                        extrinsic = gen.get_extrinsic_camera(pose, radius=radius, target=at, up=up,)
                    else:
                        radius = torch.ones_like(cam_dist) * gen.radius
                        extrinsic = gen.get_extrinsic_camera(pose, radius=radius)
                    camera_matrices = intrinsic, extrinsic
                    pred, = gen(latent_codes=codes, camera_matrices=camera_matrices, 
                                res=32, n_steps=32, mode='val')
                    ds_tar = nn.functional.interpolate(target, (32,32))
                    dists.append(nn.functional.mse_loss(pred, ds_tar, reduction='none').mean(-1).mean(-1).mean(-1))

            dists = torch.stack(dists)
            mindists, inds = dists.min(dim=0)
            pose = torch.stack([teacher[ind, k] for k, ind in enumerate(inds)])
            loss += nn.functional.mse_loss(student, pose.detach())
    
        else:
            pose = student


        if (e<30):
            random_points = torch.randn(1024,3, device=student.device)
            if cfg['training']['pose_regularisation'] == 'band':
                random_points[:,-1] = random_points[:,-1] * .5
            elif cfg['training']['pose_regularisation'] == 'top_sphere':
                random_points[:,-1] = random_points[:,-1].abs()
            random_points = random_points/random_points.norm(dim=-1, keepdim=True)
            point_dists = 1-pose.matmul(random_points.transpose(0,1))
            weights = nn.functional.softmax(2-point_dists, dim=0)
            min_point_dist = (point_dists * weights).sum(dim=0)
            exponent = min(3,e//10)
            prior_weight = 10**-exponent
            loss += prior_weight * min_point_dist.mean()


        intrinsic = gen.get_intrinsic_camera(batch_size=bsize)
        if not cfg['training']['exact_sphere']:
            radius = cam_dist.sigmoid().add(1e-5).pow(-1).add(-1).mul(gen.radius)
            extrinsic = gen.get_extrinsic_camera(pose, radius=radius, target=at, up=up,)
            if e<10:
                loss += nn.functional.mse_loss(cam_dist, torch.zeros_like(cam_dist))
        else:
            radius = torch.ones_like(cam_dist) * gen.radius.detach()
            extrinsic = gen.get_extrinsic_camera(pose, radius=radius)
        camera_matrices = intrinsic, extrinsic
        pred, depth_map = gen(latent_codes=codes, camera_matrices=camera_matrices,
                                        mode=mode, res=64, n_steps=64,
                                        return_depth_map=True)
        discrepancy = (1 + depth_map).pow(-1)

        ds_tar = nn.functional.interpolate(target, (64,64), mode='nearest')
        rec_loss = criterion(pred, ds_tar)
        loss += rec_loss
        
        if istrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        batch_time = time.time() - tic
        
        if align_mat is not None:
            pose = torch.bmm(align_mat.repeat(bsize,1,1), pose.unsqueeze(-1)).view(bsize,-1)
            pred_rotmats = gen.get_extrinsic_camera(pose)[:,:3,:3]
            gt_pose = theta2sphere(torch.stack([azimuth, elevation], dim=1).float()/180*np.pi)
            gt_rotmats = gen.get_extrinsic_camera(gt_pose)[:,:3,:3]
            avg_acc.update(angle_accuracy(pred_rotmats, gt_rotmats).cpu().item(), n=bsize)
            med_err.update(rel_angle(pred_rotmats, gt_rotmats).cpu().detach())
        if logger:
            logger.log_code("posex", pose[:,0])
            logger.log_code("posey", pose[:,1])
            logger.log_code("posez", pose[:,2])
            logger.log_code("dist", radius[:,0])
            logger.log_code("az", azimuth)
            logger.log_code("el", elevation)
        
        avg_loss_rec.update(rec_loss.cpu().item(), n=bsize)
        if i>0:
            avg_batch_time.update(batch_time)



        if i % cfg['training']['print_interval'] == 0:
            with torch.no_grad():
                camera_matrices = intrinsic[:1], extrinsic[:1]
                pred, depth_map, world_map = gen(latent_codes=codes[:1], camera_matrices=camera_matrices, 
                    mode='val', res=128, n_steps=128, return_depth_map=True, return_world_map=True)
                discrepancy = (1 + depth_map).pow(-1)
                discrepancy = nn.functional.interpolate(discrepancy, (pred.size(-2), pred.size(-1))).repeat(1,3,1,1)
                norm = (world_map.norm(dim=1)<1).float().repeat(1,3,1,1)
                norm = nn.functional.interpolate(norm, (pred.size(-2), pred.size(-1)))
                target = nn.functional.interpolate(target, (pred.size(-2), pred.size(-1)))
            save_reconstruct('epoch_'+str(e),
                            [target[0], c_inp[0], discrepancy[0], pred[0]],
                            path=exp_dir+'/plots/'+split)
            print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Depth {dmin:.4f} - {dmax:.4f}\t'.format(
                   split, i,
                   len(data), batch_time=avg_batch_time, loss=avg_loss_rec, dmin=depth_map[0].min().item(), dmax=depth_map[0].max().item()))

    if logger:
        logger.print_code(name='latest_'+split)
        if (e+1)%cfg['logs']['save_frequency']==0:
            logger.print_code(name='epoch_'+str(e+1)+'_'+split)
        logger.reset_codes()
    print('===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg acc {acc.avg:.4f}\t'
          'Med err {err.val:.4f}\n'.format(
           batch_time=int(avg_batch_time.sum), loss=avg_loss_rec, acc=avg_acc, err=med_err))



    return avg_acc.avg, med_err.val

def train(cfg):

    # define model, loss, optim
    model = models.nets.ViewNeRF(cfg['model'], device=device)
    criterion = models.loss.PerceptualLoss(losstype='mse', resize=False)
    optimizer = torch.optim.Adam([{'params': model.parameters()},])

    align_mat = torch.eye(3)

    batch_size = cfg['training']['batch_size']
    dataset = get_dataset(cfg['data'])
    train_dataset, val_dataset, test_dataset = get_loader(dataset=dataset, batch_size=batch_size)

    model = model.to(device)
    criterion = criterion.to(device)
    align_mat = align_mat.to(device)

    best_ep = 0
    best_acc = 0
    best_err = np.inf
    best_val_err = np.inf

    ckpt = cfg['training']['resume_from_ckpt']
    if ckpt:
        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict, strict=True)
        align_mat = align_preds(val_dataset, model, device, samples=100)

    logger = Code_logger(exp_dir+'/logs')
    mod_logger = Model_logger(model, exp_dir+'/ckpts')



    for i in range(cfg['training']['epochs']):

        print("=================\n=== EPOCH "+str(i+1)+" =====\n=================\n")


        # Training epoch
        train_acc, train_err = epoch(i, train_dataset, model, criterion, optimizer, logger, split="train", align_mat=align_mat, cfg=cfg)
        # Validation epoch
        align_mat = align_preds(val_dataset, model, device)
        val_acc, val_err = epoch(i, val_dataset, model, criterion, optimizer, logger, split="val", align_mat=align_mat, cfg=cfg)
        # Testing epoch
        test_acc, test_err = epoch(i, test_dataset, model, criterion, optimizer, logger, split="test", align_mat=align_mat, cfg=cfg)

        create_views(val_dataset, model, device, name='epoch_'+str(i), path=exp_dir+'/plots/gifs')

        mod_logger.dump('latest')
        if (i+1)%cfg['logs']['save_frequency']==0:
            mod_logger.dump(str(i+1))
     
        if val_err < best_val_err:
            best_val_err = val_err
            best_acc = test_acc
            best_err = test_err
            best_ep = i+1
            mod_logger.dump('best_mod')

        
        print('Best epoch: {best_ep}\t'
              'Best accuracy: {best_acc:.2f}\t'
              'Best error: {best_err:.2f}\n\n\n'.format(
                best_ep=best_ep, best_acc=best_acc*100, best_err=best_err))



def init_exps_dirs(path):
    start_time = time.gmtime()
    exp_dir = path+'exp_{:0>2}_{:0>2}_{:0>2}_{:0>2}:{:0>2}:{:0>2}'.format(
                   str(start_time.tm_year),
                   str(start_time.tm_mon),
                   str(start_time.tm_mday),
                   str(start_time.tm_hour),
                   str(start_time.tm_min),
                   str(start_time.tm_sec))
    print('Creating log directory at '+exp_dir)
    os.makedirs(exp_dir)
    os.makedirs(exp_dir+'/logs')
    os.makedirs(exp_dir+'/plots/train')
    os.makedirs(exp_dir+'/plots/val')
    os.makedirs(exp_dir+'/plots/test')
    os.makedirs(exp_dir+'/plots/gifs')
    os.makedirs(exp_dir+'/ckpts')
    return exp_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to configuration file')

    args = parser.parse_args()
    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)

    global device
    dev = cfg['training']['device']
    if torch.cuda.is_available() and isinstance(dev, int) and dev>=0:
        print('Using GPU '+str(dev))
        device = torch.device('cuda', dev)
    else:
        print('Using CPU')
        device = torch.device('cpu')

    global exp_dir
    exp_dir = init_exps_dirs(cfg['logs']['base_dir'])
    shutil.copy(args.config, exp_dir+'/conf.yml')

    train(cfg)
