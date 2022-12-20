import os
import torch
from torch.utils.data import Dataset
import torchvision

from PIL import Image
import numpy as np
import random


class ShapeNet_SRN_paired_dataset(Dataset):

    def __init__(self, split, category, src_dir,
                transform=torchvision.transforms.ToTensor(),
                views_per_model=5,
                select_topviews=False):


        self.src_dir = src_dir
        self.transform = transform
        assert split in ['train', 'val', 'test']

        self.samples_dir = os.path.join(src_dir, category+'s_'+split)

        self.objects = sorted(os.listdir(self.samples_dir))#[:64]
        self.views_per_model = views_per_model
        if select_topviews:
            image_list = []
            for obj_id in self.objects:
                topview_path = os.path.join(self.samples_dir, obj_id,'topviews_list.txt')
                if os.path.exists(topview_path):
                    with open(topview_path, 'r') as topviews_file:
                        topviews = topviews_file.read().splitlines()
                    image_list.append(random.sample(topviews, self.views_per_model))
                else:
                    views = os.listdir(os.path.join(self.samples_dir,obj_id,'rgb'))
                    topviews = [view for view in views if np.loadtxt(os.path.join(self.samples_dir, obj_id, 'pose', view.replace('png','txt')))[-5]>0]
                    print('writing views for', obj_id)
                    with open(topview_path, 'w') as topviews_file:
                        for view in topviews:
                            topviews_file.write(view+"\n")
                    image_list.append(random.sample(topviews, self.views_per_model))
        else:
            image_list = [random.sample(os.listdir(os.path.join(self.samples_dir,obj_id,'rgb')), self.views_per_model) for obj_id in self.objects]
        self.images = sorted([obj+'/rgb/'+im for obj_id, obj in enumerate(self.objects) for im in image_list[obj_id]])
        self.len = len(self.images)

        # DUPLICATE car train 4cce557de0c31a0e70a43c2d978e502e

    def __getids__(self):
        return self.ids

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        view_id = idx%self.views_per_model
        obj_id = idx//self.views_per_model
        extrinsic = torch.tensor(np.loadtxt(os.path.join(self.samples_dir, self.images[idx].replace('png','txt').replace('rgb','pose')))).view(4,4)
        intrinsic = torch.tensor(np.loadtxt(os.path.join(self.samples_dir, self.images[idx].replace('png','txt').replace('rgb','intrinsics')))).view(3,3)
        img_name = os.path.join(self.samples_dir, self.images[idx])
        pose_im = Image.open(img_name).convert('RGB')

        buddy_id = np.random.randint(self.views_per_model)
        while self.views_per_model>1 and buddy_id==view_id:
            buddy_id = np.random.randint(self.views_per_model)
        img_name = os.path.join(self.samples_dir, self.images[idx - view_id + buddy_id])
        cont_im = Image.open(img_name).convert('RGB')

        if self.transform:
            cont_im = self.transform(cont_im)
            pose_im = self.transform(pose_im)

        tar_im = pose_im


        R = extrinsic[:3,:3]
        T = extrinsic[:3,3]
        camera_position = -T
        camera_dir = camera_position/camera_position.norm()
        azimuth = -torch.atan2(camera_dir[0], camera_dir[1])/np.pi*180
        azimuth = (azimuth+360 if azimuth<0 else azimuth) - 180
        elevation = -torch.asin(camera_dir[2])/np.pi*180

        return {'pose_im': pose_im, 'cont_im': cont_im, 'tar_im': tar_im, 'azimuth': azimuth, 'elevation': elevation, 'pose_mat': extrinsic, 'K': intrinsic, 'idx':idx, 'obj_id': obj_id}


