import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import pandas as pd
# from backbone.ResNet2 import resnet_left,resnet_right
# from backbone.model2 import resnet34
# from backbone.vgg2 import vgg_left224,vgg_right224
from backbone.mix_backbone import MixNetVR
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = max_out
        x1 = self.conv1(max_out)
        return self.sigmoid(x1)*x


#
#
# class VAMM_NEW(nn.Module):
#     def __init__(self, channel, dilation_level=[1,2,3], reduce_factor=3):
#         super(VAMM_NEW, self).__init__()
#         self.planes = channel
#         self.dilation_level = dilation_level
#         self.conv = nn.Conv2d(channel, channel, kernel_size=1,padding=0)
#         self.branches = nn.ModuleList([
#                 nn.Conv2d(channel, channel, stride=1,kernel_size=1,padding=0, dilation=d) for d in dilation_level
#                 ])
#         ### ChannelGate
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Sequential(nn.Conv2d(channel,channel,kernel_size=3,padding=1),
#                                  nn.BatchNorm2d(channel),
#                                  nn.ReLU(inplace=True)
#                                  )
#                                  #(channel, channel, 1, 1, 0, bn=True, relu=True)
#         self.fc2 = nn.Conv2d(channel, (len(self.dilation_level) + 1) * channel, 1, 1, 0, bias=False)
#         self.fuse = nn.Sequential(nn.Conv2d(channel,channel,kernel_size=3,padding=1),
#                                  nn.BatchNorm2d(channel),
#                                  nn.ReLU(inplace=True)
#                                  )
#             #convbnrelu(channel, channel, k=1, s=1, p=0, relu=False)
#         ### SpatialGate
#         self.convs = nn.Sequential(
#             nn.Conv2d(channel, channel // reduce_factor, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channel // reduce_factor),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
#         )#   convbnrelu(channel, channel // reduce_factor, 1, 1, 0, bn=True, relu=True),
#                # nn.Conv2d(channel // reduce_factor, channel // reduce_factor, kernel_size=1,padding=0,stride=1, dilation=2),
#               #  nn.Conv2d(channel // reduce_factor, channel // reduce_factor, kernel_size=1,padding=0,stride=1, dilation=4),
#              #   nn.Conv2d(channel // reduce_factor, 1, 1, 1, 0, bias=False)
#                 #)
#
#     def forward(self, x):
#         conv = self.conv(x)
#         # print(conv.shape)
#         brs = [branch(conv) for branch in self.branches]
#         brs.append(conv)
#         d = self.gap(conv)
#         d = self.fc2(self.fc1(d))
#         d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)
#         s = self.convs(conv).unsqueeze(1)
#         # for i in brs:
#         #     print(i.shape)
#         # gather = sum(brs)
#         # # print(gather.shape)
#         # ### ChannelGate
#         # d = self.gap(gather)
#         # # print(d.shape)
#         # # print(self.fc1(d).shape)
#         # d = self.fc2(self.fc1(d))
#         # # print('dddd',d.shape)
#         # d = torch.unsqueeze(d, dim=1).view(-1, len(self.dilation_level) + 1, self.planes, 1, 1)
#         #
#         # ### SpatialGate
#         # s = self.convs(gather).unsqueeze(1)
#
#         ### Fuse two gates
#         f = d * s
#         f = F.softmax(f, dim=1)
#         # print('fff',f.shape)
#         return self.fuse(sum([brs[i] * f[:, i, ...] for i in range(len(self.dilation_level) + 1)]))	* x


# class PyramidPooling(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(PyramidPooling, self).__init__()
#         hidden_channel = int(in_channel / 4)
#         self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)
#
#     def forward(self, x):
#         size = x.size()[2:]
#         # print(F.adaptive_avg_pool2d(x, 1).shape)
#         # print('self.conv1(F.adaptive_avg_pool2d(x, 1))',self.conv1(F.adaptive_avg_pool2d(x, 1)).shape)
#         feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
#         feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
#         feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
#         feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
#         # print(F.adaptive_avg_pool2d(x, 1).shape)
#         # print('feat1',feat1.shape)
#         x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
#         # print('xxx',x.shape)
#         x = self.out(x)
#
#         return x

# class PyramidPooling2(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(PyramidPooling2, self).__init__()
#         hidden_channel = int(in_channel / 4)
#         # hidden_channel = in_channel
#         self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
#         # self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)
#
#     def forward(self, x):
#         size = x.size()[2:]
#         # print(F.adaptive_avg_pool2d(x, 1).shape)
#         # print('self.conv1(F.adaptive_avg_pool2d(x, 1))',self.conv1(F.adaptive_avg_pool2d(x, 1)).shape)
#         feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
#         feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
#         feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
#         feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
#         # print(F.adaptive_avg_pool2d(x, 1).shape)
#         # print('feat1',feat1.shape)
#         # x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
#         # # print('xxx',x.shape)
#         # x = self.out(x)
#
#         return feat1, feat2, feat3, feat4


# class ASPP(nn.Module):
#     def __init__(self, in_channel, depth):
#         super(ASPP, self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
#         self.conv = nn.Conv2d(in_channel, depth, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
#         self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
#         self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
#
#     def forward(self, x):
#         size = x.shape[2:]
#
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#
#         atrous_block1 = self.atrous_block1(x)
#         atrous_block6 = self.atrous_block6(x)
#         atrous_block12 = self.atrous_block12(x)
#         atrous_block18 = self.atrous_block18(x)
#
#         net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
#                                               atrous_block12, atrous_block18], dim=1))
#         return net


#
# class merger1(nn.Module):
#     def __init__(self,inchannel):
#         super(merger1, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0), nn.BatchNorm2d(inchannel), nn.ReLU(inplace=True))
#             # nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0)
#         self.conv2 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0), nn.BatchNorm2d(inchannel), nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0), nn.BatchNorm2d(inchannel), nn.ReLU(inplace=True))
#     def forward(self, rgb,dep):
#         rgb = self.conv1(rgb)
#         dep = self.conv2(dep)
#         mid = torch.multiply(rgb,dep)
#         s1 = mid +rgb
#         s2 = mid + dep
#         s3 = s1 + s2
#         out = self.conv3(s3)
#         return out


def calc_corr(a, b):
    a_avg = a.mean()
    b_avg = b.mean()

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = ((a-a_avg) * (b - b_avg)).sum()
    # cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(((a-a_avg) ** 2).sum() * ((b-b_avg) ** 2).sum())
    # sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq
    return corr_factor


class merge_corr(nn.Module):
    def __init__(self,inchannel,is_con,is_up):
        super(merge_corr, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1),
            nn.BatchNorm2d(inchannel//2),
            nn.ReLU(inplace=True)
        )

        self.is_con = is_con
        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.is_up = is_up


    def forward(self, x, y):
        # y1 = self.fc2(self.fc1(self.gap(x)))
        # y2 = self.dconv2(self.dconv1(x))
        # out = y1 * y2
        # x = torch.softmax(out,dim=1) * out
        if self.is_con == 1:
            x = self.conv1(x)
        elif self.is_con == 0:
            y= self.conv1(y)
        else:
            pass

        if self.is_up:
            x = self.up2(x)
        else:
            y = self.up2(y)

        # x1 = x.view(-1)#.contiguous()
        # print(x1.shape)
        # x1 = x.contiguous().view(x.size(0) * x.size(1), -1)
        # y1 = y.contiguous().view(y.size(0) * y.size(1), -1)
        # y1 = y.view(-1)#.contiguous()
        # x2 = pd.Series(x1.detach())
        # y2 = pd.Series(y1.detach())
        # corr = x2.corr(y2)
        # print(corr)
        corr = calc_corr(x,y)
        x_new1 = torch.multiply(x , corr)
        # y_new1 = torch.sigmoid(y)
        y_new2 = torch.tanh(y)

        y_new3 = torch.multiply(y_new2 ,(1-corr))
        x_new2 = x_new1 + y_new3
        x_new3 = torch.tanh(x_new2)
        y_new4 = torch.matmul(y_new2, x_new3)
        # y_new4 = torch.multiply(y_new2, x_new3)
        y_new4 = torch.sigmoid(y_new4)
        out = y_new4 + x + y
        # print()
        return out

class merger3(nn.Module):#channel and spatial
    def __init__(self,inchannel,is_more):
        super(merger3, self).__init__()

        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel*2, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel * 3, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel//2 , inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        self.SpatialAttention=SpatialAttention()
        # self.ChannelAttention = ChannelAttention(inchannel)
        self.more = is_more
        self.up2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
            #convbnrelu(inchannel,inchannel)
        # self.conv2 =

    def forward(self, *x):
        x1 = x[0]
        x2 = x[1]
        x11 = self.SpatialAttention(x1)
        y11 = self.SpatialAttention(x2)
        # y33 = self.SpatialAttention(x3)

        x_c = x11 * x1
        y_c = y11 * x2
        if  self.more:
            x3 = x[2]
            # print('x3', x3.shape)
            x3 = self.up2(self.conv1(x3))
            y33 = self.SpatialAttention(x3)
            xy = torch.cat((x_c, y_c,y33), dim=1)
            # print('xy',xy.shape)
            out = self.conv3(xy)

        else:
            # x3 = 0
            # y33 = x3
            xy = torch.cat((x_c, y_c), dim=1)
            # print('xy',xy.shape)
            out = self.conv2(xy)



        # print(out.shape)

        return out


class merger_AM(nn.Module):#channel and spatial
    def __init__(self,inchannel,is_more):
        super(merger_AM, self).__init__()
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.is_more=is_more
        self.conv = nn.Sequential(nn.Conv2d(inchannel*2, inchannel , kernel_size=3, padding=1),
                                  nn.BatchNorm2d(inchannel ),
                                  nn.ReLU(inplace=True))
        # self.up
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = nn.Sequential(nn.Conv2d(inchannel, inchannel // 2, kernel_size=3, padding=1),
        #                           nn.BatchNorm2d(inchannel // 2),
        #                           nn)
        #
        # self.ChannelA = ChannelAttention(inchannel)
        # self.ChannelS = SpatialAttention()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True))
            #convbnrelu(inchannel,inchannel)
        # self.conv2 =

    def forward(self, *x):
        # temp = x.mul(self.ChannelA(x))
        # temp = temp.mul(self.ChannelS(temp))
        # x = x + temp
        # temp1 = self.ChannelA(x)
        # temp2 = self.ChannelS(y)
        # temp3 =temp1+temp2+x +y
        # print('temp1',temp1.shape)
        # print('temp2', temp2.shape)
        # print('temp3', temp3.shape)
        # temp1 = self.conv1(temp3)
        # merge2 = temp3 + temp1 + temp2
        x1 = x[0]
        x2 = x[1]
        x_mid = x2
        weight_x = self.gmp(x_mid)
        x_out = x1 * weight_x
        x_outt = torch.sigmoid(x_out)
        weight_y = self.gap(x1)
        y_out = x_mid * weight_y
        y_outt = torch.sigmoid(y_out)
        if self.is_more:
            x3 = x[2]
            x3 = self.upsample2(self.conv(x3))
            final_1 = x3 * x_outt
            final_2 = x3 * y_outt
            out = final_1 + final_2
        else:
            # x3=0
            out = x_outt + y_outt
        # print('x_out',x_out.shape)
        # print('y_out',y_out.shape)
        # + x1 + x_mid+ x3

        return out


class merger5(nn.Module):
    def __init__(self,inchannel,is_more,dilation_level=[1,2,3]):
        super(merger5, self).__init__()
        # self.branches = nn.ModuleList([
        #     nn.Conv2d(inchannel, inchannel, stride=1, kernel_size=1, padding=0,dilation=d) for d in dilation_level
        # ])
        # self.branches1 = nn.ModuleList([
        #     nn.Conv2d(inchannel, inchannel, stride=1, kernel_size=3, padding=1, dilation=1),
        #     nn.Conv2d(inchannel, inchannel, stride=1, kernel_size=3, padding=2, dilation=2),
        #     nn.Conv2d(inchannel, inchannel, stride=1, kernel_size=3, padding=4, dilation=4),
        # ])
        # # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(inchannel, inchannel, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True)
        # )
        self.conv12 = nn.Sequential(
            nn.Conv2d(inchannel // 2, inchannel, kernel_size=1, padding=0),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True))
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True)
        # )
        # self.gap=nn.AdaptiveAvgPool2d(1)
        self.more = is_more
        self.up2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # self.conv3_1 = nn.Sequential(
        #     nn.Conv2d(3*inchannel, inchannel, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(inchannel),
        #     nn.ReLU(inplace=True)
        # )
    def forward(self, *x):
        rgb = x[0]
        dep = x[1]
        #
        # xr = rgb#self.conv1(rgb)
        # branch = [branch(xr) for branch in self.branches1]
        # _,x_max= torch.max(dep,dim=1)
        # x_mean = torch.mean(dep,dim=1)
        # print('dep', dep.shape)
        # print('x_max',x_max.shape)
        # print('branch[0]', branch[0].shape)
        # print('branch[1]', branch[1].shape)
        # print('branch[2]', branch[2].shape)
        # out1 = x_max.unsqueeze(1) *branch[0]+x_mean.unsqueeze(1)*branch[1]+torch.matmul(branch[2],dep)
        # out1 = x_max.unsqueeze(1) * branch[0] + x_mean.unsqueeze(1) * branch[1]
        # b,c,h,w = branch[2].shape
        # branch3 = branch[2].view(b,-1,w)
        # dep3 = dep.view(b,-1,w).permute(0,2,1)
        # # print('branch3',branch3.shape)
        # # print('dep3', dep3.shape)
        # out3 = torch.sigmoid(torch.matmul(dep3 , branch3))
        # out1 = out1 +out3.unsqueeze(1)
        # print('branch[2].shape',branch[2].shape)
        # print('dep',dep.shape)
        # out1 = out1
        # for i in branch:
        #     print(i.shape)
        # bra_cat = torch.cat((branch[0],branch[1],branch[2]),dim=1)
        # xr2 = self.conv3_1(bra_cat)
        # bra_cat =branch[0]+ branch[1]+ branch[2]
        # xr2 = bra_cat#self.conv4_1(bra_cat)
        # print('xr2',xr2.shape)
        # xd = dep#self.conv1(dep)

        # branch = [branch(xd) for branch in self.branches1]
        # bra_cat_d = torch.cat((branch[0], branch[1], branch[2]), dim=1)
        # xd2 = self.conv3_1(bra_cat_d)
        if  self.more:
            x3 = x[2]
            # print('x3', x3.shape)
            x3 = self.up2(self.conv12(x3))
            # branch = [branch(x3) for branch in self.branches1]
            # bra_cat_d = torch.cat((branch[0], branch[1], branch[2]), dim=1)
            # xr3 = self.conv3_1(bra_cat_d)
            # out = out1 + xd2+ xr2
            out = rgb+dep+ x3
        else:

            out = rgb +dep #+ xr2



        return out



class cross_level(nn.Module):
    def __init__(self,inchannel):
        super(cross_level, self).__init__()
        self.gmp=nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.conv = nn.Sequential(nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1),
        #                           nn.BatchNorm2d(inchannel//2),
        #                           nn.ReLU(inplace=True)
        #                           )
        self.dconv1 = nn.Conv2d(inchannel , inchannel, kernel_size=3, padding=2, stride=1, dilation=2)
        self.dconv2 =   nn.Conv2d(inchannel, inchannel, kernel_size=3,padding=4,stride=1, dilation=4)
        self.fc1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, kernel_size=5, padding=2),
                                 nn.BatchNorm2d(inchannel),
                                 nn.ReLU(inplace=True)
                                 )
        # (channel, channel, 1, 1, 0, bn=True, relu=True)
        self.fc2 = nn.Conv2d(inchannel, inchannel, 1, 1, 0, bias=False)
        # self.softMax = torch.softmax()
        # self.conv = nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1)

        # self.is_change = is_change
    def forward(self, x1):
        y1 = self.fc2(self.fc1(self.gap(x1)))
        y2 = self.dconv2(self.dconv1(x1))
        out = y1 * y2
        out = torch.softmax(out,dim=1) * out

        return out

class Edge_aware(nn.Module):
    def __init__(self):
        super(Edge_aware, self).__init__()
        self.sa = SpatialAttention()
        #nn.Conv2d(128, 64, kernel_size=3, padding=4, stride=1, dilation=4)
        # self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True)
                                 )
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)
                                   )
        self.last_bound = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # (channel, channel, 1, 1, 0, bn=True, relu=True)

        # self.softMax = torch.softmax()
        # self.conv = nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1)

        # self.is_change = is_change

    def forward(self, x1,x2,x3):
        x2 =self.up2(self.conv1(x2))
        x3 = self.up4(self.conv2(x3))
        x_all = torch.cat((x1,x2,x3),dim=1)
        # print('x_all',x_all.shape)
        x_all_1 = self.conv3(x_all)
        ed = self.sa(x_all_1)
        out = self.last_bound(ed)

        return out
class Edge_aware_dialation(nn.Module):
    def __init__(self):
        super(Edge_aware_dialation, self).__init__()
        self.sa = SpatialAttention()
        #
        # self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=2, stride=1, dilation=2),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True)
                                 )
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=4, stride=1, dilation=4),
            # nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(192, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True)
                                   )
        self.last_bound = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # (channel, channel, 1, 1, 0, bn=True, relu=True)

        # self.softMax = torch.softmax()
        # self.conv = nn.Conv2d(inchannel,inchannel//2,kernel_size=3,padding=1)

        # self.is_change = is_change

    def forward(self, x1,x2,x3):
        x2 =self.up2(self.conv1(x2))
        x3 = self.up4(self.conv2(x3))
        x_all = torch.cat((x1,x2,x3),dim=1)
        # print('x_all',x_all.shape)
        x_all_1 = self.conv3(x_all)
        ed = self.sa(x_all_1)
        out = self.last_bound(ed)

        return out




class SFNet(nn.Module):
    def __init__(self):
        super(SFNet, self).__init__()
        self.rgb_encode = MixNetVR(pretrained=True)
        self.dep_encode = MixNetVR(pretrained=True)
        # self.vgg = vgg_left224()
        # self.vgg_depth = vgg_right224()
        self.merge1 = merger5(64, 0)  # merger5(64)
        self.merge2 = merger5(128, 1)  # merger5(64)
        self.merge3 = merger5(256, 1)  # merger3(128)

        # self.merge1 = merger3(64,0)# merger5(64)
        # self.merge2 = merger3(128,1)#merger5(64)
        # self.merge3 = merger3(256,1)#merger3(128)
        # self.merge4 = merger5(256)#merge2()
        # self.merge5 = merger5(512)#merge2()
        self.merge4 = merger_AM(128,1)  # merge2()
        self.merge5 = merger_AM(256,1)  # merge2()
        self.merge6 = merger_AM(512,0)

        self.S5 = merge_corr(512,1,1)
        self.S4 = merge_corr(256,1,1)
        self.S3 = merge_corr(256, 0,1)
        self.S2 = merge_corr(128, 2,0)
        self.S1 = merge_corr(128, 0,0)
        # self.Edge_aware = Edge_aware()
        self.Edge_aware_dialation =  Edge_aware_dialation()
        self.conv512_128 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True)
        )
        self.conv256_64_rgb = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64),            nn.ReLU(inplace=True)
        )
        self.conv256_64_dep = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True)
                                         )
        self.conv256_128 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                         nn.ReLU(inplace=True)
                                         )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.last_conv1 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, padding=0))#, nn.BatchNorm2d(1))
        self.last_conv2 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, padding=0))#, nn.BatchNorm2d(1))
        self.last_conv3 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, padding=0))#, nn.BatchNorm2d(1))
        self.last_conv4 = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, padding=0))#, nn.BatchNorm2d(1))
        self.last_conv5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1, padding=0))  # , nn.BatchNorm2d(1))
        self.last_conv6 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, padding=0))
    def forward(self, left, depth):
        A1,A2,A3,A4,A5,A6 = self.rgb_encode(left)
        A1_d, A2_d, A3_d,A4_d, A5_d, A6_d = self.dep_encode(depth)
        merge1 = self.merge1(A1,A1_d)
        # print('merge1', merge1.shape)
        # print('A2',A2.shape)
        # print('A2_d', A2_d.shape)
        merge2 = self.merge2(A2, A2_d,merge1)
        # print('merge2', merge2.shape)
        merge3 = self.merge3(A3, A3_d,merge2)
        # print('merge3',merge3.shape)
        # print('A4_d', A4_d.shape)

        merge6 = self.merge6(A6,A6_d)
        merge5 = self.merge5(A5, A5_d,merge6)  # torch.Size([4, 512, 14, 14])
        merge4 = self.merge4(A4, A4_d,merge5)  # torch.Size([4, 512, 28, 28])

        out  = self.S5(merge6,merge5)
        # print('out',out.shape) #[4, 256, 14, 14]
        out1 = self.S4(out,merge4)
        # print('out1',out1.shape)#([4, 128, 28, 28])
        # print('merge3',merge3.shape)

        out2 = self.S3(out1,merge3)
        # print('out2', out2.shape) #out2 torch.Size([4, 128, 56, 56])
        # print('merge2', merge2.shape)#merge2 torch.Size([4, 128, 112, 112])
        out22= out2 + self.up2(out1)
        out3 = self.S2(merge2,out22)
        # print('out3', out3.shape)#4, 128, 112, 112
        # print('merge1', merge1.shape)#[4, 64, 224, 224]
        out33 = out3 + self.conv256_128(self.up8(out))
        out4 = self.S1(merge1,out33)
        # print('out1', out1.shape)
        # print('out2', out2.shape)
        # print('out3', out3.shape)
        # print('out4', out4.shape)
        # print('out', out.shape)
        # out5 = self.S4(out4, f5)
        # out = self.up4(self.last_conv1(out5))
        p0 = self.up16(self.last_conv6(out))
        p1 = self.up8(self.last_conv2(out1))
        p2 = self.up4(self.last_conv3(out2))
        # print('s3',out3.shape)
        p3 = self.up2(self.last_conv4(out3))
        p4 = self.last_conv5(out4)
        # bound_out = self.Edge_aware(merge1, merge2, merge3)
        bound_out = self.Edge_aware_dialation(merge1, merge2, merge3)
        return  p4,p0,p1,p2,p3,bound_out
        # print('input',input.shape)#[4, 384, 56, 56]






        #
        # print("x_l",x_l.shape)
        # print("x_depth_l", x_depth_l.shape)
        # print("lf1", lf1.shape)
        # print("rf1", rf1.shape)
        # print("lf2", lf2.shape)
        # print("rf2", rf2.shape)
        # print("lf3", lf3.shape)
        # print("rf3", rf3.shape)
        # print("lf4", lf4.shape)
        # print("rf4", rf4.shape)
        # print("merge1", merge1.shape)
        # print("merge2", merge2.shape)
        # print("merge3", merge3.shape)
        # print("merge4", merge4.shape)
        # print("merge5", merge5.shape)

        # if self.training == True:
        #     return out
        # else:
        #     return out





        # print("lf5",lf5.shape)

        # if self.training:
        #     return out, out1, out2, out3, out4, out5
        # return out


'''
tensorboard --logdir=/home/sunfan/1212121212/pth/SSSFF7/summary
'''
# class RegLSTM(nn.Module):
#     def __init__(self):
#         super(RegLSTM,self).__init__()
#         self.rnn = nn.LSTM(2,10,batch_first=True)
#         self.reg = nn.Sequential(
#             nn.Linear(224,1)
#         )
#     def forward(self,x):
#         x, (ht,ct)=self.rnn(x)
#         seq_len, batch_szie, hidden_size = x.shape
#         x = x.view(-1,hidden_size)
#         x = self.reg(x)
#         x = x.view(seq_len,batch_szie,-1)
#         return x
if __name__ == '__main__':
    image = torch.randn(4, 3, 224, 224)
    ndsm = torch.randn(4, 3, 224, 224)
    ndsm1 = torch.randn(1, 3, 224, 224)
    # net = merge()
    # out = net(image,ndsm,ndsm1)
    # print(out.shape)
    # net = merger5(3)
    # out = net(image,ndsm)
    net = SFNet()
    out = net(image,ndsm)
    # from FLOP import CalParams
    # CalParams(net,image,ndsm)
    # out = net(image, ndsm)
    # print(out.shape)
    for i in out:
        print(i.shape)






