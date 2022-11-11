import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torch.nn import functional as F
import numpy as np

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class B2_VGG(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self):
        super(B2_VGG, self).__init__()
        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(6, 64, 3, 1, 1))
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

        conv4_1 = nn.Sequential()
        conv4_1.add_module('pool3_1', nn.AvgPool2d(2, stride=2))
        conv4_1.add_module('conv4_1_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4_1.add_module('relu4_1_1', nn.ReLU())
        conv4_1.add_module('conv4_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('relu4_2_1', nn.ReLU())
        conv4_1.add_module('conv4_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_1.add_module('relu4_3_1', nn.ReLU())
        self.conv4_1 = conv4_1

        conv5_1 = nn.Sequential()
        conv5_1.add_module('pool4_1', nn.AvgPool2d(2, stride=2))
        conv5_1.add_module('conv5_1_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_1_1', nn.ReLU())
        conv5_1.add_module('conv5_2_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_2_1', nn.ReLU())
        conv5_1.add_module('conv5_3_1', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_1.add_module('relu5_3_1', nn.ReLU())
        self.conv5_1 = conv5_1

        conv4_2 = nn.Sequential()
        conv4_2.add_module('pool3_2', nn.AvgPool2d(2, stride=2))
        conv4_2.add_module('conv4_1_2', nn.Conv2d(256, 512, 3, 1, 1))
        conv4_2.add_module('relu4_1_2', nn.ReLU())
        conv4_2.add_module('conv4_2_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_2.add_module('relu4_2_2', nn.ReLU())
        conv4_2.add_module('conv4_3_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4_2.add_module('relu4_3_2', nn.ReLU())
        self.conv4_2 = conv4_2

        conv5_2 = nn.Sequential()
        conv5_2.add_module('pool4_2', nn.AvgPool2d(2, stride=2))
        conv5_2.add_module('conv5_1_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_2.add_module('relu5_1_2', nn.ReLU())
        conv5_2.add_module('conv5_2_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_2.add_module('relu5_2_2', nn.ReLU())
        conv5_2.add_module('conv5_3_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv5_2.add_module('relu5_3_2', nn.ReLU())
        self.conv5_2 = conv5_2

        pre_train = model_zoo.load_url(model_urls['vgg16'])
        self._initialize_weights(pre_train)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.conv4_1(x)
        x1 = self.conv5_1(x1)
        x2 = self.conv4_2(x)
        x2 = self.conv5_2(x2)
        return x1, x2

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv1.conv1_1.weight.data.copy_([pre_train[keys[0]], pre_train[keys[0]]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
        self.conv4_1.conv4_1_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4_1.conv4_2_1.weight.data.copy_(pre_train[keys[16]])
        self.conv4_1.conv4_3_1.weight.data.copy_(pre_train[keys[18]])
        self.conv5_1.conv5_1_1.weight.data.copy_(pre_train[keys[20]])
        self.conv5_1.conv5_2_1.weight.data.copy_(pre_train[keys[22]])
        self.conv5_1.conv5_3_1.weight.data.copy_(pre_train[keys[24]])
        self.conv4_2.conv4_1_2.weight.data.copy_(pre_train[keys[14]])
        self.conv4_2.conv4_2_2.weight.data.copy_(pre_train[keys[16]])
        self.conv4_2.conv4_3_2.weight.data.copy_(pre_train[keys[18]])
        self.conv5_2.conv5_1_2.weight.data.copy_(pre_train[keys[20]])
        self.conv5_2.conv5_2_2.weight.data.copy_(pre_train[keys[22]])
        self.conv5_2.conv5_3_2.weight.data.copy_(pre_train[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(np.array([pre_train[keys[0]], pre_train[keys[0]]]))
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
        self.conv4_1.conv4_1_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4_1.conv4_2_1.bias.data.copy_(pre_train[keys[17]])
        self.conv4_1.conv4_3_1.bias.data.copy_(pre_train[keys[19]])
        self.conv5_1.conv5_1_1.bias.data.copy_(pre_train[keys[21]])
        self.conv5_1.conv5_2_1.bias.data.copy_(pre_train[keys[23]])
        self.conv5_1.conv5_3_1.bias.data.copy_(pre_train[keys[25]])
        self.conv4_2.conv4_1_2.bias.data.copy_(pre_train[keys[15]])
        self.conv4_2.conv4_2_2.bias.data.copy_(pre_train[keys[17]])
        self.conv4_2.conv4_3_2.bias.data.copy_(pre_train[keys[19]])
        self.conv5_2.conv5_1_2.bias.data.copy_(pre_train[keys[21]])
        self.conv5_2.conv5_2_2.bias.data.copy_(pre_train[keys[23]])
        self.conv5_2.conv5_3_2.bias.data.copy_(pre_train[keys[25]])


# class vgg16(nn.Module):
#     def __init__(self):
#         super(vgg16, self).__init__()
#         self.cfg = {'tun': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 'tun_ex': [512, 512, 512]}
#         self.extract = [8, 15, 22, 29] # [3, 8, 15, 22, 29]
#         self.extract_ex = [5]
#         self.base = nn.ModuleList(vgg(self.cfg['tun'], 3))
#         self.base_ex = vgg_ex(self.cfg['tun_ex'], 512)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, 0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def load_pretrained_model(self, model):
#         self.base.load_state_dict(model)
#
#     def forward(self, x, multi=0):
#         tmp_x = []
#         for k in range(len(self.base)):
#             x = self.base[k](x)
#             if k in self.extract:
#                 tmp_x.append(x)
#         x = self.base_ex(x)
#         tmp_x.append(x)
#         if multi == 1:
#             tmp_y = []
#             tmp_y.append(tmp_x[0])
#             return tmp_y
#         else:



class VGG(nn.Module):
    # VGG16 with two branches
    # pooling layer at the front of block
    def __init__(self,pretrained =1):
        super(VGG, self).__init__()
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
            self._initialize_weights(pre_train)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x1,x2,x3,x4,x5

    def _initialize_weights(self, pre_train):
        keys = list(pre_train.keys())
        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
        self.conv4.conv4_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4.conv4_2.weight.data.copy_(pre_train[keys[16]])
        self.conv4.conv4_3.weight.data.copy_(pre_train[keys[18]])
        self.conv5.conv5_1.weight.data.copy_(pre_train[keys[20]])
        self.conv5.conv5_2.weight.data.copy_(pre_train[keys[22]])
        self.conv5.conv5_3.weight.data.copy_(pre_train[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
        self.conv4.conv4_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4.conv4_2.bias.data.copy_(pre_train[keys[17]])
        self.conv4.conv4_3.bias.data.copy_(pre_train[keys[19]])
        self.conv5.conv5_1.bias.data.copy_(pre_train[keys[21]])
        self.conv5.conv5_2.bias.data.copy_(pre_train[keys[23]])
        self.conv5.conv5_3.bias.data.copy_(pre_train[keys[25]])


PretrainPath = 'PretrainModel/'


def gen_convs(inchannel, outchannel, bn=False):
    yield nn.Conv2d(inchannel, outchannel, 3, padding=1)

    if bn:
        yield nn.BatchNorm2d(outchannel)

    yield nn.ReLU(inplace=True)


def gen_fcs(infeature, outfeature):
    yield nn.Linear(infeature, outfeature)
    yield nn.ReLU(inplace=True)
    yield nn.Dropout(p=0.5)


class VGGNet(nn.Module):

    def __init__(self, name, cls=False, multi=False, bn=False):

        super(VGGNet, self).__init__()
        C = [3, 64, 128, 256, 512, 512]
        FC = [25088, 4096, 4096]
        N = [2, 2, 3, 3, 3] if 'vgg16' in name else [2, 2, 4, 4, 4]

        self.cls = cls
        self.multi = multi
        self.convs = nn.ModuleList(
            [nn.Sequential(*[m for j in range(N[i]) for m in gen_convs(C[min(i + j, i + 1)], C[i + 1], bn=bn)]) for i in
             range(5)])
        self.fc = nn.Sequential(*[m for i in range(2) for m in gen_fcs(FC[i], FC[i + 1])]) if cls else None

    def forward(self, X):

        features = []
        for i in range(5):
            Out = self.convs[i](X)
            features.append(Out)
            X = F.max_pool2d(Out, kernel_size=2, stride=2)

        if self.cls:
            fc = X.view(X.shape[0], -1)
            fc = self.fc(fc)

            return fc

        else:
            return features if self.multi else features[-1]

    def load(self, path):

        convs = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        fcs = [0, 3, 6]

        count = 0
        state_dict = torch.load(path, map_location='cpu')
        for conv in self.convs:
            print(len(conv))
            for i in range(0, len(conv), 2):
                conv[i].weight.data = state_dict['features.' + str(convs[count]) + '.weight']
                conv[i].bias.data = state_dict['features.' + str(convs[count]) + '.bias']
                count += 1

        count = 0
        print(len(self.fc))
        for i in range(0, len(self.fc), 3):
            self.fc[i].weight.data = state_dict['classifier.' + str(fcs[count]) + '.weight']
            self.fc[i].bias.data = state_dict['classifier.' + str(fcs[count]) + '.bias']
            count += 1


def vgg(name=None, cls=False, multi=False, pretrain=False):
    assert 'vgg16' in name or 'vgg19' in name

    vgg = VGGNet(name, cls=cls, multi=multi, bn=True if 'bn' in name else False)

    if pretrain:
        # print('It should load pre-trained VGG16, but we omit it')
        vgg.load_state_dict(torch.load('../PretrainModel/' + name + '.pkl', map_location='cpu'), strict=False)

    return vgg


# if __name__ == '__main__':
#
#     '''
#     a = vgg('vgg19', multi=True, cls=True)
#     print a
#     X = torch.empty(5, 3, 224, 224)
#     for encoder in a(X):
#         print encoder.shape
#     '''
#     '''
#     m = vgg('vgg19', cls=True, multi=True)
#     m.load('../../PretrainModel/vgg19.pth')
#     torch.save(m.state_dict(), 'vgg19.pkl')
#     '''
#     a = torch.load('vgg19.pkl', map_location='cpu')
#     b = torch.load('../../PretrainModel/vgg19.pth', map_location='cpu')
#
#     afk = sorted(filter(lambda fc: 'fc' in fc, a.keys()))
#     bfk = sorted(filter(lambda cl: 'classifier' in cl, b.keys()))
#     for af, bf in zip(afk, bfk):
#         print(af, bf)
#         assert torch.equal(a[af], b[bf])
#     '''
#     ack = sorted(filter(lambda convs : 'convs' in convs, a.keys()))
#     bck = sorted(filter(lambda convs : 'features' in convs, b.keys()), key=lambda a : int(a.split('.')[1]))
#     '''
#     ack = filter(lambda convs: 'convs' in convs, a.keys())
#     bck = filter(lambda convs: 'features' in convs, b.keys())
#     for ak, bk in zip(ack, bck):
#         print(ak, bk)
#         assert torch.equal(a[ak], b[bk])


if __name__ == '__main__':
    from blocks.FLOP import CalParams
    rgb = torch.randn((1, 6, 224, 224)).cuda()
    depth = torch.randn(1, 3, 224, 224).cuda()
    model = B2_VGG().cuda()
    # CalParams(model,rgb,depth)
    outputs = model(rgb)
    print([i.size() for i in outputs])