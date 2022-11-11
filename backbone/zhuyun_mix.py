from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import torchvision.models as models
import torch
from blocks.GCnet_weight import ContextBlock2d
from blocks.gcnet import GC_layers
from mmcv.cnn import constant_init, kaiming_init
from collections import OrderedDict
import re

model = models.densenet121(pretrained=True)
model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

model2 = models.resnet50(pretrained=True)
model2_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class MDRGnet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3],  pretrained=True, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,):
        super(MDRGnet, self).__init__()
        # resnet_encode_rgb
        self.inplanes = 64
        block = Bottleneck
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #  resnet_decode层
        self.decode_layer5_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )  # 64*224*224
        self.decode_layer4_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer3_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer2_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.convrgb = nn.Conv2d(64, 1, 1)
        self.decode_layer1_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )

        # resnet_conv+upsample
        self.uplayer5 = nn.Sequential(
            nn.Conv2d(2048, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear'), )
        self.uplayer4 = nn.Sequential(
            nn.Conv2d(1024, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear'), )
        self.uplayer3 = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear'), )
        self.uplayer2 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), )
        self.uplayer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), )

        # rgb_gc
        self.gc_r5 = GC_layers(2048)
        self.gc_r4 = GC_layers(1024)
        self.gc_r3 = GC_layers(512)
        self.gc_r2 = GC_layers(256)
        self.gc_r1 = GC_layers(64)

        # rgb_Supervision
        self.sv4 = self.convd = nn.Conv2d(64, 1, 1)
        self.sv3 = self.convd = nn.Conv2d(64, 1, 1)
        self.sv2 = self.convd = nn.Conv2d(64, 1, 1)

        # desnet_encode_depth
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        # vggnet_decode层
        self.decode_layer5_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )  # 64*224*224
        self.decode_layer4_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer3_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer2_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.convd = nn.Conv2d(64, 1, 1)
        self.decode_layer1_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )

        # vgg_conv+upsample
        self.uplayer5_d = nn.Sequential(
            nn.Conv2d(1024, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear'), )
        self.uplayer4_d = nn.Sequential(
            nn.Conv2d(1024, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear'), )
        self.uplayer3_d = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear'), )
        self.uplayer2_d = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), )
        self.uplayer1_d = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), )

        # depth_gc
        self.gc_d5 = GC_layers(1024)
        self.gc_d4 = GC_layers(1024)
        self.gc_d3 = GC_layers(512)
        self.gc_d2 = GC_layers(256)
        self.gc_d1 = GC_layers(64)

        # depth_Supervision
        self.sv4_d = self.convd = nn.Conv2d(64, 1, 1)
        self.sv3_d = self.convd = nn.Conv2d(64, 1, 1)
        self.sv2_d = self.convd = nn.Conv2d(64, 1, 1)



        if pretrained:
            pretrained_dense = model_zoo.load_url(model_urls['densenet121'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_dense.items():

                # model_dict[k.replace('features.0.w','features_deep.0.w')] = t.mean(v,1).data.view_as(state_dict[k.replace('features.0.w','features_deep.0.w')])
                if k in state_dict:
                    # print(k)
                    model_dict[k] = v
                    # model_dict[k[:8]+'_deep'+k[8:]] = v

            pretrained_res = model_zoo.load_url(model2_urls['resnet50'])
            for k, v in pretrained_res.items():
                if k in state_dict:
                    # print(k)
                    # model_dict[k.replace('features.0.w','features_deep.0.w')] = t.mean(v,1).data.view_as(state_dict[k.replace('features.0.w','features_deep.0.w')])
                    model_dict[k] = v
                # model_dict[k[:8]+'_deep'+k[8:]] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.Dropout2d(),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, rgb, depth):
        # rgb
        A1 = self.conv1(rgb)
        # F1 = self.gc_r1(A1)
        F1 = self.uplayer1(A1)
        A1 = self.bn1(A1)
        A1 = self.relu(A1)
        A1 = self.maxpool(A1)
        A2 = self.layer1(A1)
        A2 = self.gc_r2(A2)
        F2 = self.uplayer2(A2)
        A3 = self.layer2(A2)
        A3 = self.gc_r3(A3)
        F3 = self.uplayer3(A3)
        # print(F4.size())
        A4 = self.layer3(A3)
        A4 = self.gc_r4(A4)
        F4 = self.uplayer4(A4)
        A5 = self.layer4(A4)
        A5 = self.gc_r5(A5)
        F5 = self.uplayer5(A5)
        # print(F5.size())
        f5 = torch.cat((F5, F4), dim=1)
        f4 = self.decode_layer5_r(f5)
        s4 = f4
        f4 = torch.cat((f4, F3), dim=1)
        f3 = self.decode_layer4_r(f4)
        s3 = f3
        f3 = torch.cat((F2, f3), dim=1)
        f2 = self.decode_layer3_r(f3)
        s2 = f2
        f2 = torch.cat((F1, f2), dim=1)
        f1 = self.decode_layer2_r(f2)
        f_r = self.convrgb(f1)
        # print(f_r.size())

        # depth
        A1_d = self.features.conv0(depth)
        A1_d = self.features.norm0(A1_d)
        A1_d = self.features.relu0(A1_d)
        A2_d = self.features.pool0(A1_d)
        A2_d = self.features.denseblock1(A2_d)
        A3_d = self.features.transition1(A2_d)
        A3_d = self.features.denseblock2(A3_d)
        A4_d = self.features.transition2(A3_d)
        A4_d = self.features.denseblock3(A4_d)
        A5_d = self.features.transition3(A4_d)
        A5_d = self.features.denseblock4(A5_d)

        f5_d = torch.cat((F5_d, F4_d), dim=1)
        f4_d = self.decode_layer5_d(f5_d)
        s4_d = f4_d
        f4_d = torch.cat((f4_d, F3_d), dim=1)
        f3_d = self.decode_layer4_d(f4_d)
        s3_d = f3_d
        f3_d = torch.cat((F2_d, f3_d), dim=1)
        f2_d = self.decode_layer3_d(f3_d)
        s2_d = f2_d
        f2_d = torch.cat((F1_d, f2_d), dim=1)
        f1_d = self.decode_layer2_d(f2_d)
        f_d = self.convd(f1_d)

        # supervision
        s4 = self.sv4(s4)
        s4_d = self.sv4_d(s4_d)
        out4 = s4 + s4_d
        s3 = self.sv3(s3)
        s3_d = self.sv3_d(s3_d)
        out3 = s3 + s3_d
        s2 = self.sv2(s2)
        s2_d = self.sv2_d(s2_d)
        out2 = s2 + s2_d


        out = f_r + f_d
        if self.training:
            return out, out4, out3, out2
        return out

# 辣鸡
class MRDGnet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3],  pretrained=True, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,):
        super(MRDGnet, self).__init__()
        # resnet_encode_rgb
        self.inplanes = 64
        block = Bottleneck
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #  resnet_decode层
        self.decode_layer5_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )  # 64*224*224
        self.decode_layer4_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer3_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer2_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.convrgb = nn.Conv2d(64, 1, 1)
        self.decode_layer1_r = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )

        # resnet_conv+upsample
        self.uplayer5 = nn.Sequential(
            nn.Conv2d(2048, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear'), )
        self.uplayer4 = nn.Sequential(
            nn.Conv2d(1024, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear'), )
        self.uplayer3 = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear'), )
        self.uplayer2 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), )
        self.uplayer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), )

        # rgb_gc
        self.gc_r5 = GC_layers(2048)
        self.gc_r4 = GC_layers(1024)
        self.gc_r3 = GC_layers(512)
        self.gc_r2 = GC_layers(256)
        self.gc_r1 = GC_layers(64)

        # rgb_Supervision
        self.sv4 = self.convd = nn.Conv2d(64, 1, 1)
        self.sv3 = self.convd = nn.Conv2d(64, 1, 1)
        self.sv2 = self.convd = nn.Conv2d(64, 1, 1)

        # desnet_encode_depth
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2


        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))


        # vggnet_decode层
        self.decode_layer5_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )  # 64*224*224
        self.decode_layer4_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer3_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.decode_layer2_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )
        self.convd = nn.Conv2d(64, 1, 1)
        self.decode_layer1_d = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(), )

        # vgg_conv+upsample
        self.uplayer5_d = nn.Sequential(
            nn.Conv2d(1024, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear'), )
        self.uplayer4_d = nn.Sequential(
            nn.Conv2d(1024, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=16, mode='bilinear'), )
        self.uplayer3_d = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=8, mode='bilinear'), )
        self.uplayer2_d = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), )
        self.uplayer1_d = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'), )

        # depth_gc
        self.gc_d5 = GC_layers(1024)
        self.gc_d4 = GC_layers(1024)
        self.gc_d3 = GC_layers(512)
        self.gc_d2 = GC_layers(256)
        self.gc_d1 = GC_layers(64)

        # depth_Supervision
        self.sv4_d = self.convd = nn.Conv2d(64, 1, 1)
        self.sv3_d = self.convd = nn.Conv2d(64, 1, 1)
        self.sv2_d = self.convd = nn.Conv2d(64, 1, 1)



        if pretrained:
            pretrained_dense = model_zoo.load_url(model_urls['densenet121'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_dense.items():

                # model_dict[k.replace('features.0.w','features_deep.0.w')] = t.mean(v,1).data.view_as(state_dict[k.replace('features.0.w','features_deep.0.w')])
                if k in state_dict:
                    # print(k)
                    model_dict[k] = v
                    # model_dict[k[:8]+'_deep'+k[8:]] = v

            pretrained_res = model_zoo.load_url(model2_urls['resnet50'])
            for k, v in pretrained_res.items():
                if k in state_dict:
                    # print(k)
                    # model_dict[k.replace('features.0.w','features_deep.0.w')] = t.mean(v,1).data.view_as(state_dict[k.replace('features.0.w','features_deep.0.w')])
                    model_dict[k] = v
                # model_dict[k[:8]+'_deep'+k[8:]] = v

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.Dropout2d(),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, rgb, depth):
        # rgb
        A1_d = self.features.conv0(rgb)
        A1_d = self.features.norm0(A1_d)
        A1_d = self.features.relu0(A1_d)
        F1_d = self.uplayer1_d(A1_d)
        A2_d = self.features.pool0(A1_d)
        A2_d = self.features.denseblock1(A2_d)
        F2_d = self.gc_d2(A2_d)
        F2_d = self.uplayer2_d(F2_d)
        A3_d = self.features.transition1(A2_d)
        A3_d = self.features.denseblock2(A3_d)
        F3_d = self.gc_d3(A3_d)
        F3_d = self.uplayer3_d(F3_d)
        A4_d = self.features.transition2(A3_d)
        A4_d = self.features.denseblock3(A4_d)
        F4_d = self.gc_d4(A4_d)
        F4_d = self.uplayer4_d(F4_d)
        A5_d = self.features.transition3(A4_d)
        A5_d = self.features.denseblock4(A5_d)
        F5_d = self.gc_d5(A5_d)
        F5_d = self.uplayer5_d(F5_d)
        f5_d = torch.cat((F5_d, F4_d), dim=1)
        f4_d = self.decode_layer5_d(f5_d)
        s4_d = f4_d
        f4_d = torch.cat((f4_d, F3_d), dim=1)
        f3_d = self.decode_layer4_d(f4_d)
        s3_d = f3_d
        f3_d = torch.cat((F2_d, f3_d), dim=1)
        f2_d = self.decode_layer3_d(f3_d)
        s2_d = f2_d
        f2_d = torch.cat((F1_d, f2_d), dim=1)
        f1_d = self.decode_layer2_d(f2_d)
        f_d = self.convd(f1_d)

        # depth
        A1 = self.conv1(depth)
        # F1 = self.gc_r1(A1)
        F1 = self.uplayer1(A1)
        A1 = self.bn1(A1)
        A1 = self.relu(A1)
        A1 = self.maxpool(A1)
        A2 = self.layer1(A1)
        A2 = self.gc_r2(A2)
        F2 = self.uplayer2(A2)
        A3 = self.layer2(A2)
        A3 = self.gc_r3(A3)
        F3 = self.uplayer3(A3)
        # print(F4.size())
        A4 = self.layer3(A3)
        A4 = self.gc_r4(A4)
        F4 = self.uplayer4(A4)
        A5 = self.layer4(A4)
        A5 = self.gc_r5(A5)
        F5 = self.uplayer5(A5)
        # print(F5.size())
        f5 = torch.cat((F5, F4), dim=1)
        f4 = self.decode_layer5_r(f5)
        s4 = f4
        f4 = torch.cat((f4, F3), dim=1)
        f3 = self.decode_layer4_r(f4)
        s3 = f3
        f3 = torch.cat((F2, f3), dim=1)
        f2 = self.decode_layer3_r(f3)
        s2 = f2
        f2 = torch.cat((F1, f2), dim=1)
        f1 = self.decode_layer2_r(f2)
        f_r = self.convrgb(f1)
        # print(f_r.size())

        # supervision
        s4 = self.sv4(s4)
        s4_d = self.sv4_d(s4_d)
        out4 = s4 + s4_d
        s3 = self.sv3(s3)
        s3_d = self.sv3_d(s3_d)
        out3 = s3 + s3_d
        s2 = self.sv2(s2)
        s2_d = self.sv2_d(s2_d)
        out2 = s2 + s2_d


        out = f_r + f_d
        if self.training:
            return out, out4, out3, out2
        return out



if __name__ == '__main__':
    # model = VRnet()
    #
    # inputs = torch.randn((1, 3, 224, 224))
    import torch as t
    rgb = t.randn(1, 3, 224, 224)
    depth = t.randn(1, 3, 224, 224)
    # net = ContextBlock2d(inplanes=128, planes=2048)
    # out = net(rgb)
    net = MRDGnet()
    out = net(rgb, depth)
    # print(out)
    for i in out:
        print(i.shape)
