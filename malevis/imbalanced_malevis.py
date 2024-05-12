#dataset resample result
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import torch.nn.functional as F
from collections import Counter


def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 9100 / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

per_cls = get_img_num_per_cls(26, imb_type='exp', imb_factor=0.1)
print(per_cls)
total_sum = sum(per_cls)
print(total_sum)