# import torchvision.models as models
# import torch.nn as nn
# # https://pytorch.org/docs/stable/torchvision/models.html#id3
#
import torch


from torchvision import models
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
from torch.autograd import Variable

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MixNetVR(nn.Module):
    def __init__(self, block=BasicBlock, layers=[4, 6, 3], pretrained=True):
        super(MixNetVR, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))

        self.conv1 = conv1
        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3
        pre_train = model_zoo.load_url(model_urls['vgg16'])
        self._initialize_weights(pre_train)
        # conv1 = nn.Sequential()
        # conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        # conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        # conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        # conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        #
        # self.conv1 = conv1
        # conv2 = nn.Sequential()
        # conv2.add_module('pool1', nn.AvgPool2d(2, stride=2))
        # conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        # conv2.add_module('relu2_1', nn.ReLU())
        # conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        # conv2.add_module('relu2_2', nn.ReLU())
        # self.conv2 = conv2
        #
        # conv3 = nn.Sequential()
        # conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        # conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        # conv3.add_module('relu3_1', nn.ReLU())
        # conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        # conv3.add_module('relu3_2', nn.ReLU())
        # conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        # conv3.add_module('relu3_3', nn.ReLU())
        # self.conv3 = conv3

        self.inplanes = 64
        self.layer2 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[1], stride=2)  # 6
        self.layer4 = self._make_layer(block, 512, layers[2], stride=2)  # 3
        self.conv256_64 = nn.Sequential(nn.Conv2d(256, 64, (3, 3), (1, 1),padding=1), nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))


        if pretrained:
            # pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            self._initialize_weights(pre_train)

            pretrained_res = model_zoo.load_url(model_urls['resnet34'])
            for k, v in pretrained_res.items():
                if k in state_dict:
                    model_dict[k] = v



        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # A1 = self.conv1(x)
        # A2 = self.conv2(A1)
        # A3 = self.conv3(A2)
        # A3_1 = self.conv256_64(A3)
        A1 = self.conv1(x)
        A2 = self.conv2(A1)
        A3 = self.conv3(A2)
        A3_1 = self.conv256_64(A3)
        # print('A3',A3.shape)
        # print('A3_1', A3_1.shape)
        A4 = self.layer2(A3_1)
        A5 = self.layer3(A4)
        A6 = self.layer4(A5)

        return A1,A2,A3,A4,A5,A6


class MixNetRV(nn.Module):
    def __init__(self, block=BasicBlock, layers=[4, 6, 3], pretrained=True):
        super(MixNetRV, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),  # [6:13]
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),  # [13:23]
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),  # [13:33]
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, 1, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),  # [33:43]
        #
        # )

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3
        self.conv256_128 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True)
                                         )
        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.AvgPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3_1', nn.AvgPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('relu4_3', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4_1', nn.AvgPool2d(2, stride=2))
        conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_1', nn.ReLU())
        conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_2', nn.ReLU())
        conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
        conv5.add_module('relu5_3', nn.ReLU())
        self.conv5 = conv5
        pre_train = model_zoo.load_url(model_urls['vgg16'])
        if pretrained:
            # pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            self._initialize_weights(pre_train)
            model_dict = {}
            state_dict = self.state_dict()
            # for k, v in pretrained_vgg.items():
            #     if k in state_dict:
            #         print('k', k)
            #         print('v', v.shape)
            #         model_dict[k] = v

            pretrained_res = model_zoo.load_url(model_urls['resnet34'])
            for k, v in pretrained_res.items():
                if k in state_dict:
                    model_dict[k] = v

            state_dict.update(model_dict)
            self.load_state_dict(state_dict)
    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv4.conv4_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4.conv4_2.weight.data.copy_(pre_train[keys[16]])
        self.conv4.conv4_3.weight.data.copy_(pre_train[keys[18]])
        self.conv5.conv5_1.weight.data.copy_(pre_train[keys[20]])
        self.conv5.conv5_2.weight.data.copy_(pre_train[keys[22]])
        self.conv5.conv5_3.weight.data.copy_(pre_train[keys[24]])

        self.conv4.conv4_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4.conv4_2.bias.data.copy_(pre_train[keys[17]])
        self.conv4.conv4_3.bias.data.copy_(pre_train[keys[19]])
        self.conv5.conv5_1.bias.data.copy_(pre_train[keys[21]])
        self.conv5.conv5_2.bias.data.copy_(pre_train[keys[23]])
        self.conv5.conv5_3.bias.data.copy_(pre_train[keys[25]])









    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x1)
        # print('x',x.shape)
        x1 = self.layer1(x)
        # print('x1', x1.shape)
        x2 = self.layer2(x1)
        # print('x2', x2.shape)
        x3 = self.layer3(x2)
        # print('x3', x3.shape)
        # print('x3',x3.shape)
        x4= self.conv256_128(x3)
        A1 = self.conv3(x4)
        # print('A1', A1.shape)
        A2 = self.conv4(A1)
        # print('A2', A2.shape)
        A3 = self.conv5(A2)
        # print('A3', A3.shape)
        # return x1, x2, x3, x4, x5
        # A1 = self.features[:13](x3)
        # A2 = self.features[13:23](A1)
        # A3 = self.features[23:33](A2)

        return x1,x2,x3,A1,A2,A3



if __name__ == "__main__":
    Net = MixNetRV()
    indata = torch.rand(4, 3, 224, 224)
    out = Net(indata)
    for i in out:
         print(i.shape )