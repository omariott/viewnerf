import torch.nn as nn
import torchvision.transforms as T
import torchvision.models
import numpy as np


class PerceptualLoss(nn.Module):
    def __init__(self, losstype='mse', reduction="mean", greyscale=False, resize=False):
        super(PerceptualLoss, self).__init__()

        def double_loss(output, target):
            return .5 * (nn.functional.mse_loss(output, target) + nn.functional.l1_loss(output, target))
        def cosine_loss(output, target):
            bsize = output.size(0)
            return 1 - nn.functional.cosine_similarity(output.view(bsize, -1), target.view(bsize, -1))

        self.net = torchvision.models.vgg16_bn(pretrained=True).features.eval()
        for p in self.net:
            p.requires_grad = False
#        print(self.net)
#        exit()
        """
        vgg11 : [2,5,10,15,20]
        vgg11_bn : [3,7,14,21,28]
        vgg16 : [3,8,15,22,29]
        vgg16_bn : [5,12,22,32,42]
        vgg19 : [3,8,17,26,35]
        """
        self.layers_id = [0, 5, 12, 22]#, 23, 30]
        self.weights = [10, 1, 1, 1]#, 1, 1]
        if losstype=='l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif losstype=='mse':
            self.loss = nn.MSELoss(reduction=reduction)
        elif losstype=='smooth':
            self.loss = nn.SmoothL1Loss(reduction=reduction, beta=.1)
        elif losstype=='both':
            self.loss = double_loss
        elif losstype=='cosine':
            self.loss = cosine_loss
        else:
            print("error, I don't know the loss ", losstype)
            exit()
        self.intrinsic_weight = np.array(self.weights).sum()

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.greyscale = greyscale
        self.resize = resize

    def cuda(self):
        self.net = self.net.cuda()
        return self

    def forward(self, input_data, target, pixel_weights=None):
        x = self.normalize(input_data)
        tar = self.normalize(target)
        if self.greyscale:
            x = x.mean(1, keepdim=True).repeat(1,3,1,1)
            tar = tar.mean(1, keepdim=True).repeat(1,3,1,1)
        if self.resize:
            x = nn.functional.interpolate(x, 224)
            tar = nn.functional.interpolate(tar, 224)
        else:
            tar = nn.functional.interpolate(tar, x.size(-1))
        losses = 0
        weight_id = 0
        for id, module in enumerate(self.net):
            if id in self.layers_id:
                if pixel_weights is None:
                    lossval = self.loss(x.clone(), tar.clone())
                else:
                    pixel_weights = nn.functional.interpolate(pixel_weights, (x.size(-2), x.size(-1)))
                    lossval = self.loss(pixel_weights * x.clone(), pixel_weights * tar.clone())
                losses += self.weights[weight_id] * lossval
                weight_id += 1
            if id == self.layers_id[-1]:
                break
            x = module(x)
            tar = module(tar)
        return losses/self.intrinsic_weight
