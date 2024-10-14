import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from archs import wavelet
from utils.color_change import rgb2hsi

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(

            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

# Multi-Scale Pyramid Module
class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.tanh = nn.Tanh()
        self.refine2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.refine3 = nn.Conv2d(16 + 3, 16, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest

    def forward(self, x):
        dehaze = self.relu((self.refine2(x)))
        shape_out = dehaze.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(dehaze, 128)

        x102 = F.avg_pool2d(dehaze, 64)

        x103 = F.avg_pool2d(dehaze, 32)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze


class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self, x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):

        if x.shape[1] == 3:
            x = self.get_gray(x)
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=1, band_kernel_size=7, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        # self.instance_1 = nn.InstanceNorm2d(10, affine=True)
        # self.instance_1 = nn.InstanceNorm2d(10, affine=True)
        self.instance = nn.InstanceNorm2d(in_channels, affine=True)
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        # return self.instance(torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1, ))
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1, )

class SplitAug(nn.Module):
    def __init__(self, inc=3, bsc=16):
        super().__init__()

        # self.conv_x = nn.Conv2d(bsc//8, bsc//4, kernel_size=1, stride=1, padding=0)
        # self.conv_y = nn.Conv2d(bsc//8, bsc//4, kernel_size=1, stride=1, padding=0)

        self.meanpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.torch_sobel = GradLayer()

        self.conv_first_a = nn.Conv2d(bsc // 8, bsc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_b = nn.Conv2d(bsc // 8, bsc, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_id = nn.Conv2d(inc, bsc, kernel_size=3, stride=1, padding=1, bias=False)

        self.instance_a = nn.InstanceNorm2d(bsc, affine=True)
        self.instance_b = nn.InstanceNorm2d(bsc, affine=True)
        self.instance_id = nn.InstanceNorm2d(bsc, affine=True)

        self.conv_out_a = nn.Conv2d(bsc, bsc // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_b = nn.Conv2d(bsc, bsc // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_id = nn.Conv2d(bsc, bsc // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv2d(bsc, bsc, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.Tanh()
        self.pa = PALayer(bsc)

        self.lk = InceptionDWConv2d(bsc)
    def forward(self, x):

        edg = self.torch_sobel(x)
        edg_r = self.torch_sobel(x[:, 0:1, :, :])
        edg_g = self.torch_sobel(x[:, 1:2, :, :])
        edg_b = self.torch_sobel(x[:, 2:3, :, :])
        edg_c = torch.cat([edg_r, edg_g, edg_b], dim=1)
        t = self.meanpool(edg_c)  # 4 3 1 1
        max_indices = torch.argmin(t, dim=1).squeeze(0).squeeze(0)  # 4 1 1
        b1 = []
        b2 = []
        for i in range(x.shape[0]):
            a = torch.cat([edg[i], x[i, max_indices[i]:max_indices[i]+1, :, :]], dim=0)
            b = torch.cat(([x[i, :max_indices[i], :, :], x[i, max_indices[i]+1:, :, :]]), dim=0)
            b1.append(a)
            b2.append(b)
        a = torch.stack(b1, dim=0)
        b = torch.stack(b2, dim=0)
        first_a = self.conv_first_a(a)
        first_b = self.conv_first_b(b)
        id = self.conv_id(x)

        first_a = self.instance_a(first_a)
        first_b = self.instance_b(first_b)
        id = self.instance_id(id)

        out_a = self.conv_out_a(first_a)
        out_b = self.conv_out_b(first_b)
        out_id = self.conv_out_id(id)

        mix = out_a + out_b + out_id
        # out_instance = self.conv_out(self.pa(self.conv_out(torch.cat([out_a, out_b, out_id, mix], dim=1))))
        out_instance = self.lk(self.conv_out(torch.cat([out_a, out_b, out_id, self.act(mix)], dim=1)))

        return out_instance

class Pool_x(nn.Module):
    def __init__(self, inplanes=16, outplanes=16, norm_layer=nn.BatchNorm2d):
        super(Pool_x, self).__init__()
        midplanes = outplanes // 2
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1)) #shuchu gaodu h kuandu wei 1
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # [4, 16, 128, 128]
        _, _, h, w = x.size()

        x1 = self.pool1(x)  # [4, 16, 128, 1]
        x1 = self.conv1(x1)  # [4, 8, 128, 1]
        x1 = self.bn1(x1)  # [4, 8, 128, 1]
        x1 = x1.expand(-1, -1, h, w)  # [4, 8, 128, 128]
        x1 = self.relu(x1)  # [4, 8, 128, 128]
        weight = self.conv3(x1).sigmoid()  # [4, 16, 128, 128]
        out = weight * x + x  # [4, 8, 128, 128]
        return out


class Pool_y(nn.Module):
    def __init__(self, inplanes=16, outplanes=16, norm_layer=nn.BatchNorm2d):
        super(Pool_y, self).__init__()
        midplanes = outplanes // 2
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = norm_layer(midplanes)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()
        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x2 = self.relu(x2)
        weight = self.conv3(x2).sigmoid()
        out = x * weight + x
        return out

# Multi-Scale Pyramid Module
class JiaoHu(nn.Module):
    def __init__(self, org_channel=3, base_nf=16):
        super(JiaoHu, self).__init__()

        self.refine1 = nn.Conv2d(org_channel, base_nf, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(base_nf, base_nf, kernel_size=3, stride=1, padding=1)
        # self.refine2 = nn.Conv2d(base_nf, base_nf, kernel_size=1, stride=1, padding=0)
        self.enh_x = Pool_x()
        self.enh_y = Pool_y()
        self.act = nn.ReLU()
        # self.bn = nn.BatchNorm2d(base_nf)

    def forward(self, lh, hl, hh):

        lh_end = self.enh_x(self.refine1(lh))
        hl_end = self.enh_y(self.refine1(hl))
        hh_1 = self.refine1(hh)
        hh_end = self.refine2(self.act((hh_1 + hh_1 * (lh_end + hl_end).sigmoid())))

        return lh_end, hl_end, hh_end


class Fusion(nn.Module):
    def __init__(self, n_feats=16):
        super().__init__()
        # i_feats =n_feats*2

        self.Conv1 = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.Tanh(),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))
        self.Conv1_1 = nn.Sequential(
            nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0),
            nn.Tanh(),
            nn.Conv2d(2 * n_feats, n_feats, 1, 1, 0))

        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.FF = FreBlock()

        self.pyd = Enhance()
        self.tanh = nn.Tanh()

    def forward(self, ll, lh, hl, hh):
        ll_f = self.pyd(ll)
        mix = lh + hl + hh
        b, c, H, W = mix.shape
        mix_mag, mix_pha = self.FF(mix)
        # Ghost Expand
        mix_mag = self.Conv1(mix_mag)
        mix_pha = self.Conv1_1(mix_pha)
        real_main = mix_mag * torch.cos(mix_pha)
        imag_main = mix_mag * torch.sin(mix_pha)
        x_out_main = torch.complex(real_main, imag_main)
        x_out_main = torch.abs(torch.fft.irfft2(x_out_main, s=(H, W), norm='backward')) + 1e-8

        l_w = self.tanh(x_out_main).sigmoid()
        h_w = self.tanh(ll_f).sigmoid()

        ll_o = (ll_f * l_w + ll_f)
        lh_o = (h_w * lh + lh)
        hl_o = (h_w * hl + hl)
        hh_o = (h_w * hh + hh)
        out = torch.cat([ll_o, lh_o, hl_o, hh_o], dim=0)

        return out


class FreBlock(nn.Module):
    def __init__(self):
        super(FreBlock, self).__init__()

    def forward(self, x):
        x = x + 1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)

        return mag, pha



class WB(nn.Module):
    def __init__(self, bsc=16):
        super().__init__()
        self.conv = nn.Conv2d(bsc, bsc, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.conv1 = nn.Conv2d(bsc, bsc, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(bsc, bsc, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(bsc, bsc, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(bsc, bsc, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.avgpool(x)  # 10 3 1 1
        out2 = self.conv(x)  # 10 3 1 1
        out3 = self.maxpool(x)
        out4 = self.conv1(x)
        wh = self.conv2(out2 / (out1 + 1e-5))
        wh_weight = self.tanh(wh).sigmoid()
        gray = self.conv3(out4 / (out3 + 1e-5))
        gray_weight = self.tanh(gray).sigmoid()
        out = self.conv4(x + wh_weight * x + gray_weight * x)
        return out

# MAIN-Net
class Enlight(nn.Module):
    def __init__(self, out_nc=3, base_nf=16):
        super(Enlight, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # self.encoder = Condition()
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.stage2 = PALayer(base_nf)
        self.act = nn.ReLU(inplace=True)
        self.wt = wavelet.DWT()
        self.saug = SplitAug()
        self.jiaohu = JiaoHu()
        self.fusion = Fusion()
        self.iwt = wavelet.IWT()
        self.wb = WB()
        self.tanh = nn.Tanh()
    def cc_enl(self, x):
        t = rgb2hsi(x)
        _, i = torch.split(t, [2, 1], dim=1)
        y = (1-i) * x + x
        minhs = -self.maxpool(-y)
        maxhs = self.maxpool(y)
        y = (y - minhs) / (maxhs - minhs + 1e-5)
        return y

    def forward(self, x):
        x = self.cc_enl(x)
        identy = self.tanh(self.saug(x))
        x_ll, x_lh, x_hl, x_hh = self.wt(x)
        x_en_ll = self.saug(x_ll)
        x_en_lh, x_en_hl, x_en_hh = self.jiaohu(x_lh, x_hl, x_hh)
        fus = self.fusion(x_en_ll, x_en_lh, x_en_hl, x_en_hh)
        out_stage = self.iwt(fus) + identy

        out_stage = self.act(out_stage)
        out_vgg = self.conv4(out_stage)
        out_stage = self.conv2(out_stage)

        out_stage = self.wb(out_stage)

        out_stage = self.stage2(out_stage)

        out_stage = self.act(out_stage)
        out_stage = self.conv3(out_stage)
        return out_stage, out_vgg
