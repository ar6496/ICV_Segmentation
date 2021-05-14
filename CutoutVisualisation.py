# -*- coding: utf-8 -*-
"""
@author: kshit
"""
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
np.random.seed(23)
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = 1
        self.length = 16

    def transform_cutout(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.ToTensor())
train_dataset = datasets.CIFAR10(root='data/',train=True,transform=train_transform,download=True)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Cutout Regularization')
image,_ = train_dataset[0]
t = Cutout(1, 16)
kk = t.transform_cutout(image)
ax1.imshow(kk.permute(1, 2, 0))
image,_ = train_dataset[1]
t = Cutout(1, 16)
kk = t.transform_cutout(image)
ax2.imshow(kk.permute(1, 2, 0))
plt.show()
