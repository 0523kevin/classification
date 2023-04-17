"""
data_utils.py
"""

import os
import glob
import logging
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset


def init_transform(name, p):
    transform_dict = {
        'grayscale': torch.nn.Sequential(
                    transforms.Grayscale(num_output_channels=3),
                ),
        'gaussian': transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur((5,9), sigma=(0.1, 5))]), p=0.5),
        'gray_crop': torch.nn.Sequential(transforms.RandomApply(torch.nn.ModuleList([
                                    transforms.Grayscale(num_output_channels=3),
                                    # transforms.ColorJitter(0.5),
                                    # transforms.RandomHorizontalFlip(p=1),
                                    transforms.CenterCrop(size=(320,320)),
                                    transforms.Resize((512,384)),
                                    # transforms.GaussianBlur((5,9), sigma=(0.1,5)),
                                ]), p=p)),

    # 'gray_hor_crop_gau1': torch.nn.Sequential(transforms.RandomApply(torch.nn.ModuleList([
    #                                         transforms.Grayscale(num_output_channels=3),]), p=0.5),
	# 									# transforms.RandomApply(torch.nn.ModuleList([
    #                                     #     transforms.ColorJitter(0.5),]), p=0.5),
	# 									# transforms.RandomApply(torch.nn.ModuleList([
    #                                     #     transforms.RandomHorizontalFlip(p=1),]), p=0.5),
    #                                     transforms.RandomApply(torch.nn.ModuleList([
    #                                         transforms.CenterCrop(size=(320,320))]), p=0.1),
    #                                     # transforms.RandomApply(torch.nn.ModuleList([
    #                                         # transforms.GaussianBlur((5,9), sigma=(0.1,5))]), p=0.5)),
    }
    return transform_dict[name]


def generate_file_list(data_dir, val_split=0.2, train=True):
    """Generate a list of image files.
    Args:
        data_dir (str):   
    Returns:

    """
    query = os.path.abspath(f'{data_dir}/**/*.jpg')
    image_files = glob.glob(query, recursive=True)
    image_files.sort()

    train_size = int(len(image_files) * (1 - val_split))
    # file_indices = np.arange(len(image_files))

    if train:
        image_files = image_files[:train_size]
    else:
        image_files = image_files[train_size:]        

    if not image_files:
        logging.warning(f"No image file found in '{data_dir}'.")

    return image_files


class MaskDataset(Dataset):
    def __init__(self, image_files, target, group_age=True, transform=None):
        self.image_files = image_files
        self.target = target
        self.group_age = group_age
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def _gender_to_cls(self, gender):
        assert gender in ['male', 'female']

        if gender == 'male':
            return 0
        elif gender == 'female':
            return 1
        
    def _age_to_cls(self, age):
        if age < 30:
            return 0
        elif 30 <= age < 60:
            return 1
        else:
            return 2

    def _mask_to_cls(self, mask):
        # mask_types = {'incorrect_mask': 0, 'mask1': 1, 'mask2': 2,
        #               'mask3': 3, 'mask4': 4, 'mask5': 5, 'normal': 6}
        mask_types = {'incorrect_mask': 1, 'mask1': 0, 'mask2': 0,
                      'mask3': 0, 'mask4': 0, 'mask5': 0, 'normal': 2}
        
        return mask_types[mask]

    def __getitem__(self, idx):
        fpath = self.image_files[idx]
        img = Image.open(fpath)

        if self.transform:
            img = self.transform(img)

        img = transforms.ToTensor()(img)

        label = {}

        fpath_split = os.path.split(fpath)
        mask = fpath_split[1].split('.')[0]
        identity = os.path.split(fpath_split[0])[1].split('_')
        age, gender = int(identity[3]), identity[1]

        if self.group_age:
            age = self._age_to_cls(age)

        label['age'] = age
        label['gender'] = self._gender_to_cls(gender)
        label['mask'] = self._mask_to_cls(mask)

        if self.target == 'all':
            label = 6 * label['mask'] + 3 * label['gender'] + label['age']
            return img, label
        else:
            return img, label[self.target]



def get_dataloader(dataset, batch_size=8, shuffle=True, drop_last=True):
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    msg = 'Get dataloader...\n'
        
    msg += f'\tTotal Dataset size: {len(dataset)}\n'
    msg += f'\tBatch size: {batch_size}, # iters: {len(dataloader)}'

    logging.info(msg)

    return dataloader