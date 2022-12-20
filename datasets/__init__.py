import torchvision

from . import ShapeNet_SRN, FCars
import sys

transf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(128, 128)),
        torchvision.transforms.ToTensor()
        ])

def get_dataset(cfg):
    dataset_name = cfg['dataset']

    if dataset_name == 'ShapeNet_SRN':
        train_dataset = ShapeNet_SRN.ShapeNet_SRN_paired_dataset(split='train', category=cfg['category'],
            src_dir=cfg['data_dir'], transform=transf)
        val_dataset = ShapeNet_SRN.ShapeNet_SRN_paired_dataset(split='val', category=cfg['category'],
            src_dir=cfg['data_dir'], transform=transf)
        test_dataset = ShapeNet_SRN.ShapeNet_SRN_paired_dataset(split='test', category=cfg['category'],
            src_dir=cfg['data_dir'], transform=transf)

    elif dataset_name == 'Freiburg_cars':
        Fcar_dataset = FCars.Freiburg_cars_paired_dataset(src_dir=cfg['data_dir'], transform=transf)
        train_dataset, val_dataset, test_dataset = Fcar_dataset.get_splits()

    else:
        print('Dataset in configuration file must be either \'ShapeNet_SRN\' or \'Freiburg_cars\', exiting')
        sys.exit(0)

    return (train_dataset, val_dataset, test_dataset)
