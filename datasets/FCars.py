import os
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import numpy as np
from PIL import ImageOps


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

def square_bbox(bbox, border=0):
    x1, y1, x2, y2 = bbox
    maxlen = max(x2-x1, y2-y1)
    halflen = maxlen//2 + border
    xavg, yavg = (x1+x2)//2, (y1+y2)//2
    square_box = (xavg-halflen, yavg-halflen, xavg+halflen, yavg+halflen)
    return square_box


def ratio_bbox(bbox, ratio=2, safe=True):
    x1, y1, x2, y2 = bbox
    ylen = y2-y1
    halflen = ylen//2
    xlen = int(halflen*ratio)
    xavg, yavg = (x1+x2)//2, (y1+y2)//2
    ratio_box = (xavg-xlen, yavg-halflen, xavg+xlen, yavg+halflen)
    return ratio_box

def crop_to_square(image, mode='tight'):
    width, height = image.size
    if mode == 'tight':
        length = min(width, height)
    else:
        length = max(width, height)
    halflen = length//2
    xavg, yavg = width//2, height//2
    bbox = (xavg - halflen, yavg - halflen, xavg + halflen, yavg + halflen)
    return image.crop(bbox)


class Freiburg_cars_paired_dataset(Dataset):
    """Freiburg cars dataset."""

    def __init__(self, src_dir, transform=torchvision.transforms.ToTensor(), pose_transform=None, subset=None):

        
        self.src_dir = os.path.join(src_dir, "images")
        self.lab_dir = os.path.join(src_dir, "annotations")
        self.mask_dir = os.path.join(src_dir, "instance_masks")

        self.transform = transform
        if pose_transform is None:
            self.pose_transform = transform
        else:
            self.pose_transform = pose_transform

        self.ids = sorted(os.listdir(self.src_dir))
        self.images = sorted([obj_id+'/'+im_id for obj_id in self.ids for im_id in os.listdir(os.path.join(self.src_dir, obj_id))])

        self.len = len(self.images)

        self.instance_seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()


    def instance_segment_images(self, CUDA=True):

        if not os.path.exists(self.mask_dir):
            os.mkdir(self.mask_dir)

        for image in self.images:
            img_name = os.path.join(self.src_dir, image)

            mask_name = img_name.replace(self.src_dir, self.mask_dir)
            if not os.path.exists(os.path.dirname(mask_name)):
                os.mkdir(os.path.dirname(mask_name))
            elif os.path.exists(mask_name):
                continue

            image = Image.open(img_name).convert('RGB')
            h, w = image.width, image.height
            image_tensor = torchvision.transforms.ToTensor()(image)
            model = self.instance_seg_model
            if CUDA:
                image_tensor = image_tensor.cuda()
                model.cuda()
            if h*w > 1000000:
#                print("This won't fit in GPU!")
                image_tensor = image_tensor.cpu()
                model.cpu()
            out = model([image_tensor])[0]
            boxes = out['boxes']
            largest_box = np.argmax([(x2-x1) * (y2-y1) for x1, y1, x2, y2 in boxes])
            mask = out['masks'][largest_box]
            torchvision.utils.save_image(mask.float(), mask_name)

        for model in self.segmentation_models:
            model.cpu()




    def get_annotations(self, im_id):
        car_id = im_id.split('/')[-2]
        car_num = str(int(car_id[-2:]))
        img_name = car_id + '/' + im_id.split('/')[-1]
        annotations_name = os.path.join(self.src_dir, car_num+'_annot.txt').replace(self.src_dir, self.lab_dir)
        with open(annotations_name) as f:
            for line in f:
                name, x1, y1, x2, y2, az = line.split()
                if name[:-4] == img_name[:-4]:
                    break
        bbox = int(x1), int(y1), int(x2), int(y2)
        return bbox, int(az)

    def get_splits(self):
        car_inds = np.array([int(im_path[3:6]) for im_path in self.images])
        train_inds = np.where(car_inds<=45)[0]
        val_inds = np.where((car_inds>45) & (car_inds<=47))[0]
        test_inds = np.where(car_inds>47)[0]
        assert len(train_inds) + len(val_inds) + len(test_inds) == len(self)
        train_set = torch.utils.data.Subset(self, train_inds)
        val_set = torch.utils.data.Subset(self, val_inds)
        test_set = torch.utils.data.Subset(self, test_inds)
        return train_set, val_set, test_set


    def __getids__(self):
        return self.ids


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        img_name = os.path.join(self.src_dir,
                                self.images[idx])
        image = Image.open(img_name).convert('RGB')
        H, W = image.width, image.height

        car_id = img_name.split('/')[-2]
        obj_id = int(car_id[-3:])
        car_dir = os.path.join(self.src_dir, car_id)
        other_views_list = os.listdir(car_dir)
        other_views_id = np.random.randint(len(other_views_list))
        other_view_name = os.path.join(car_dir,
                                other_views_list[other_views_id])

        cont_image = Image.open(other_view_name).convert('RGB')

        mask_name = img_name.replace(self.src_dir, self.mask_dir)
        mask = Image.open(mask_name).convert('1')#
        arr_mask = np.array(Image.open(mask_name)) / 255
        arr_mask = (arr_mask > .5).astype(np.uint8)
        arr_im = np.array(image)
        segmented_im = arr_im * arr_mask + 255*(1-arr_mask)

        image = Image.fromarray(segmented_im.astype(np.uint8)) 

        bbox, az = self.get_annotations(img_name)

        pose_bbox = square_bbox(ratio_bbox(bbox, ratio=2.5))
        expanded_im = ImageOps.expand(image, 1000, (255,255,255))
        pose_image = expanded_im.crop([p+1000 for p in pose_bbox])
        tar_bbox = square_bbox(ratio_bbox(bbox, ratio=2.5))
        tar_image = expanded_im.crop([p+1000 for p in tar_bbox])
        mask = mask.crop(tar_bbox)

        ""
        cont_bbox, _ = self.get_annotations(other_view_name)
        cont_bbox = square_bbox(cont_bbox)
        cont_image = cont_image.crop(cont_bbox)
        ""

        if self.transform:
            pose_image = self.pose_transform(pose_image)
            cont_image = self.transform(cont_image)
            tar_image = self.transform(tar_image)
            mask = self.transform(mask)

        azimuth = az - 180
        elevation = 0

        az_tensor = azimuth * torch.ones(1)
        pose_mat = angles2rotmat(az_tensor,torch.zeros_like(az_tensor),torch.zeros_like(az_tensor))[0]

        return {'pose_im': pose_image, 'cont_im': cont_image, 'tar_im': tar_image, 'mask': mask, 'azimuth': azimuth, 'elevation': elevation, 'pose_mat':pose_mat, 'idx':idx, 'obj_id': obj_id}

