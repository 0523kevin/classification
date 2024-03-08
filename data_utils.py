import os
import logging
import numpy as np
from glob import glob
from enum import Enum
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from albumentations import *
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from sklearn.model_selection import train_test_split
        
def init_transform(name, p):
    transform_dict = {
        "valid" : transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        "norm" : transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        'grayscale': transforms.Grayscale(num_output_channels=3),
        'gaussian': transforms.GaussianBlur((5,9), sigma=(0.1, 5)),
        'center_crop' : transforms.Compose([
            transforms.CenterCrop(size=(320,320)),
            transforms.Resize((512,384)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]),
        'gray_crop': transforms.Compose([
            transforms.RandomApply(
                nn.ModuleList([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.CenterCrop(size=(320,320))]), p=p)
            ]),
        "train_albu" : Compose([
            Resize(512,384),
            GaussianBlur(51, (0.1, 2.0)),
            ColorJitter(brightness=0.5, saturation=0.5, hue=0.5), 
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]),
    }
    return transform_dict[name]


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2
    
    @classmethod
    def from_str(cls, value:str) -> int:
        mask_dict = {'mask1':cls.MASK,
                     'mask2':cls.MASK,
                     'mask3':cls.MASK,
                     'mask4':cls.MASK,
                     'mask5':cls.MASK,
                     'incorrect_mask':cls.INCORRECT,
                     'normal':cls.NORMAL}
        return mask_dict[value]
    
class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2
    
    @classmethod
    def from_int(cls, value:str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age label should be numeric type : {value}")
        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD
        
class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1
    
    @classmethod
    def from_str(cls, value:str) -> int:
        value = value.lower()
        if value == 'male':
            return cls.MALE
        elif value == 'female':
            return cls.FEMALE
        else:
            raise ValueError(f"Gender labels should be either male or female : {value}")
        
def get_multiclass_label(fpath):
    # ~~/data/train/images/000001_female_Asian_45/mask1.jpg -> [~~/data/train/images/000001_female_Asian_45/, mask1.jpg
    fpath_split = os.path.split(fpath)
    # mask.jpg -> mask
    mask = fpath_split[1].split('.')[0]
    identity = os.path.split(fpath_split[0])[1].split('_')
    age, gender = int(identity[3]), identity[1]
    multi_class_label = 6 * MaskLabels.from_str(mask) + 3 * GenderLabels.from_str(gender) + AgeLabels.from_int(age)
    return multi_class_label

def make_filelist(img_dir, val_size=0.2, stratify=True):
    filelist = glob(f"{img_dir}/**/*.jpg")
    if stratify:
        labels = []
        for fpath in filelist:       
            multi_class_label = get_multiclass_label(fpath)
            labels.append(multi_class_label)
            
        x_train, x_val = train_test_split(filelist, test_size=val_size, stratify=labels)
        return x_train, x_val
        
    else: 
        train_size = int(len(filelist) * (1 - val_size))
        train_image_files, val_image_files = filelist[:train_size], filelist[train_size:] 
        return train_image_files, val_image_files
        
class MaskDataset(Dataset):
    num_classes = 3 * 2 * 3
    def __init__(self, img_files, transform=None):
        super().__init__()
        self.img_files = img_files
        self.transform = transform    
                
    def set_transform(self, transform):
        self.transform = transform
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path)
        # torchvision.transforms
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        # albumentations
        # image = np.array(image)
        # if self.transform:
        #     image = self.transform(image=image)['image']
        
        label = get_multiclass_label(img_path)
        return image, label
        
    def __len__(self):
        return len(self.img_files)
    
def get_dataloader(dataset,
                   batch_size=8,
                   shuffle=True,
                   drop_last=True,
                   sampler=None):
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            sampler=sampler)
    
    msg = f"Get dataloader ... \n"
    msg += f"\tTotal Dataset size : {len(dataset)}\n"
    msg += f"\tBatch size : {batch_size}, #Iters : {len(dataloader)}"
    logging.info(msg)
    return dataloader