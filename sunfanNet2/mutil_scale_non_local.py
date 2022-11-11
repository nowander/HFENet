import torch
import torch.nn as nn
import torchvision

class NonLocalBlock(nn.Module):
    def __init__(self, channel, ratio):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // ratio
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y=None):
        # [N, C, H, W]
        b, c, h, w = x.size()
        # [N, C // ratio, H*W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C // ratio]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]M
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C // ratio]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_phi)
        # [N, C // ratio, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H, W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        if y is not None:
            y = torch.mul(x, y)
            out = out + y
        return out

class Mutil_scale_Non_Local(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(Mutil_scale_Non_Local, self).__init__()
        self.nonlocal5 = NonLocalBlock(in_channel, ratio)
        self.nonlocal4 = NonLocalBlock(in_channel, ratio)
        self.nonlocal3 = NonLocalBlock(in_channel, ratio)
        self.nonlocal2 = NonLocalBlock(in_channel, ratio)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2, x3, x4):
        y1 = self.nonlocal5(x1)
        y2 = self.nonlocal4(x2, self.upsample2(y1))
        y3 = self.nonlocal3(x3, self.upsample2(y2))
        y4 = self.nonlocal2(x4, self.upsample2(y3))
        return y4

if __name__ == '__main__':
    # model = NonLocalBlock(channel=16)
    # input = torch.randn(1, 16, 64, 64)
    # out = model(input)
    # print(out.shape)
    x1 = torch.randn(1, 128, 8, 8)
    x2 = torch.randn(1, 128, 16, 16)
    x3 = torch.randn(1, 128, 32, 32)
    x4 = torch.randn(1, 128, 64, 64)
    model = Mutil_scale_Non_Local(in_channel=128, ratio=2)
    out = model(x1, x2, x3, x4)
    print(out.shape)

