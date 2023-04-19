"""
models.py
"""

import torch
from torch import nn

import timm
# from torchvision.models import resnet34

# def init_model_dict():
#     model_dict = {
#         'model': 'Model',
#         'renet34': ''
#     }

#     return model_dict

def init_model(model_name, n_class):
    return timm.create_model(model_name, pretrained=True, num_classes=n_class)


# class ResNet50Custom(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.resnet = timm.resnet50(pretrained=True)
#         num_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         x = self.resnet(x)
#         return x
    

# class VGG16Custom(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.vgg = timm.vgg16(pretrined=True)
#         num_features = self.vgg.classifier[6].in_features
#         self.vgg.classifier[6] = nn.Linear(num_features, num_classes)

#     def forward(self, x):
#         x = self.vgg(x)
#         return x
    
# class MobileNetV2Custom(nn.Module):
#     def __init__(self, num_classe):
#         super().__init__()
#         self.mobilenet = models.mobilenet_v2(pretrained=True)
#         num_features = self.mobilenet.classifier[1].in_features
#         self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)
    
#     def forward(self, x):
#         x = self.mobilenet(x)
#         return x




# class ResNet(nn.Module):
#     def __init__(self, in_channels, nker=64, norm="bnorm", nblk=[3,4,6,3]):
#         super().__init__()

#         self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=1, bias=True, norm=None, relu=True)
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         res_1 = ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
#         self.res_1 = nn.Sequential(*[res_1 for _ in range(nblk[0])])

#         res_2 = ResBlock(nker*2, nker*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
#         self.res_2_up = ResBlock(nker, nker*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True, init_block=True)
#         self.res_2 = nn.Sequential(*[res_2 for _ in range(nblk[1]-1)])

#         res_3 = ResBlock(nker*2*2, nker*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
#         self.res_3_up = ResBlock(nker*2, nker*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True, init_block=True)
#         self.res_3 = nn.Sequential(*[res_3 for _ in range(nblk[2]-1)])

#         res_4 = ResBlock(nker*2*2*2, nker*2*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True, init_block=True)
#         self.res_4_up = ResBlock(nker*2*2, nker*2*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
#         self.res_4 = nn.Sequential(*[res_4 for _ in range(nblk[3]-1)])

#         self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
#         self.fc = nn.Linear(nker*2*2, 3)

#     def forward(self, x):
#         x = self.enc(x)
#         x = self.max_pool(x)
#         x = self.res_1(x)
#         x = self.max_pool(x)

#         x = self.res_2_up(x, short_cut=True)
#         x = self.res_2(x)
#         x = self.max_pool(x)

#         x = self.res_3_up(x, short_cut=True)
#         x = self.res_3(x)
#         # x = self.max_pool(x)

#         # x = self.res_4_up(x, short_cut=False)
#         # x = self.res_4(x)
        
#         x = self.avg_pooling(x)
#         x = x.view(x.shape[0], -1)
#         out = self.fc(x)
#         return out


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=True):
#         super().__init__()

#         layers = []
#         layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                              kernel_size=kernel_size, stride=stride, padding=padding,
#                              bias=bias)]

#         if norm == "bnorm":
#             layers += [nn.BatchNorm2d(num_features=out_channels)]

#         if relu:
#             layers += [nn.ReLU()]

#         self.cbr = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.cbr(x)


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
#                  padding=1, bias=True, norm="bnorm", short_cut=False, relu=True, init_block=False):
#         super().__init__()

#         layers = []


#         if init_block:
#           init_stride = 2
#         else:
#           init_stride = stride

#         # 1st conv
#         layers += [ConvBlock(in_channels=in_channels, out_channels=out_channels,
#                          kernel_size=kernel_size, stride=init_stride, padding=padding,
#                          bias=bias, norm=norm, relu=relu)]

#         # 2nd conv
#         layers += [ConvBlock(in_channels=out_channels, out_channels=out_channels,
#                          kernel_size=kernel_size, stride=stride, padding=padding,
#                          bias=bias, norm=norm, relu=False)]

#         self.resblk = nn.Sequential(*layers)
        
        
#         self.short_cut = nn.Conv2d(in_channels, out_channels, (1,1), stride=2)

#     def forward(self, x, short_cut=False):
#         if short_cut:
#             return self.short_cut(x) + self.resblk(x)
#         else:
#             return x + self.resblk(x) # residual connection




# # class Model(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.model_finetune = resnet34(pretrained=True)

# #     def forward(self, x):
# #         return self.model_finetune(x)


# # class Model(nn.Module):
# #     def __init__(self, n_class):
# #         super(Model, self).__init__()
# #         self.n_class = n_class

# #         self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
# #         self.conv_2 = nn.Conv2d(16, n_class, kernel_size=3, stride=1)
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
# #         self.leaky_relu = nn.LeakyReLU(0.01)
# #         self.relu = nn.ReLU()


# #     def forward(self, x):
# #         x = self.leaky_relu(self.conv_1(x))
# #         x = self.leaky_relu(self.conv_2(x))
# #         x = self.avg_pool(x)
# #         x = self.relu(x)
# #         x = x.view(-1, self.n_class)
# #         return x
    


