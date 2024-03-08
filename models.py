import torch
from torch import nn

import timm

def init_model(model_name, num_classes):
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)

import torch
from torch import nn

import timm

def init_model(model_name, num_classes):
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)
