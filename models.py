import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = self.dropout2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
    

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU()
    )

class DarkResidual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out
    
class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            conv_batch(3, 32),
            conv_batch(32, 64, stride=2),
            self.make_layer(block, in_channels=64, num_blocks=1),
            conv_batch(64, 128, stride=2),
            self.make_layer(block, in_channels=128, num_blocks=2),
            conv_batch(128, 256, stride=2),
            self.make_layer(block, in_channels=256, num_blocks=8),
            conv_batch(256, 512, stride=2),
            self.make_layer(block, in_channels=512, num_blocks=8),
            conv_batch(512, 1024, stride=2),
            self.make_layer(block, in_channels=1024, num_blocks=4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out
    
    def make_layer(block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)
    
def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)




# def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
#     return nn.Sequential(
#         nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
#         nn.BatchNorm2d(out_num),
#         nn.LeakyReLU())

# class DarkResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         reduced_channels = int(in_channels/2)

#         self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
#         self.layer2 = conv_batch(reduced_channels, in_channels)

#     def forward(self, x):
#         residual = x
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out += residual
#         return out
    

# class Darknet53(nn.Module):
#     def __init__(self, block, num_classes):
#         super().__init__()
#         self.num_classes = num_classes

#         self.features = nn.Sequential(
#             conv_batch(3, 32),
#             conv_batch(32, 64, stride=2),
#             self.make_layer(block, in_channels=64, num_blocks=1),
#             conv_batch(64, 128, stride=2),
#             self.make_layer(block, in_channels=128, num_blocks=2),
#             conv_batch(128, 256, stride=2),
#             self.make_layer(block, in_channels=256, num_blocks=8),
#             conv_batch(256, 512 stride=2),
#             self.make_layer(block, in_channels=512, num_blocks=8),
#             conv_batch(512, 1024, stride=2),
#             self.make_layer(block, in_channels=1024, num_blocks=4),
#         )
#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(1024, self.num_classes)


#     def forward(self, x):
#         out = self.features(x)
#         out = self.global_avg_pool(out)
#         out = out.view(-1, 1024)
#         out = self.fc(out)
#         return out


#     def make_layer(self, block, in_channels, num_blocks):
#         layers = []
#         for i in range(0, num_blocks):
#             layers.append(block(in_channels))
#         return nn.Sequential(*layers)
    
# def darknet53(num_classes):
#     return Darknet53(DarkResidualBlock, num_classes)

