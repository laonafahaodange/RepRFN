from torch import nn
import torch
import torch.nn.functional as F
from itertools import repeat
import collections.abc


def conv_layer(in_channels, out_channels, kernel_size, stride=1):
    # kernel_size参数预处理
    if not isinstance(kernel_size, collections.abc.Iterable):
        kernel_size = tuple(repeat(kernel_size, 2))
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        act_func = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        act_func = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    # TODO: 新增silu和gelu激活函数
    elif act_type == 'silu':
        pass
    elif act_type == 'gelu':
        pass
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return act_func


# thanks for ECBSR: https://github.com/xindongzhang/ECBSR
class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        if self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None
        tmp = self.scale * self.mask
        k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
        for i in range(self.out_planes):
            k1[i, i, :, :] = tmp[i, 0, :, :]
        b1 = self.bias
        # re-param conv kernel
        RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
        # re-param conv bias
        RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                            stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                            stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_3x3_branch_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels,
                                                    kernel_size=(1, 1),
                                                    stride=1, padding=(0, 0), dilation=1, groups=1,
                                                    padding_mode='zeros', bias=False)
            self.rbr_1x1_3x3_branch_3x3 = nn.Conv2d(in_channels=2 * in_channels, out_channels=out_channels,
                                                    kernel_size=(3, 3),
                                                    stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros',
                                                    bias=False)
            self.rbr_conv1x1_sbx_branch = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
            self.rbr_conv1x1_sby_branch = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
            self.rbr_conv1x1_lpl_branch = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)

    def forward(self, inputs):
        if (self.deploy):
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation(
                self.rbr_3x3_branch(inputs) + self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(
                    inputs) + self.rbr_1x1_branch(inputs) + self.rbr_1x1_3x3_branch_3x3(
                    self.rbr_1x1_3x3_branch_1x1(inputs)) + inputs + self.rbr_conv1x1_sbx_branch(
                    inputs) + self.rbr_conv1x1_sby_branch(inputs) + self.rbr_conv1x1_lpl_branch(inputs))

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_3x1_branch')
        self.__delattr__('rbr_1x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.__delattr__('rbr_1x1_3x3_branch_1x1')
        self.__delattr__('rbr_1x1_3x3_branch_3x3')
        self.__delattr__('rbr_conv1x1_sbx_branch')
        self.__delattr__('rbr_conv1x1_sby_branch')
        self.__delattr__('rbr_conv1x1_lpl_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data

        # 1x1 1x3 3x1 branch
        kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
                                                                                       self.rbr_1x3_branch,
                                                                                       self.rbr_3x1_branch)
        # 1x1+3x3 branch
        kernel_1x1_3x3_fuse = self._fuse_1x1_3x3_branch(self.rbr_1x1_3x3_branch_1x1,
                                                        self.rbr_1x1_3x3_branch_3x3)
        # identity branch
        device = kernel_1x1_3x3_fuse.device  # just for getting the device
        kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            kernel_identity[i, i, 1, 1] = 1.0

        kernel_1x1_sbx, bias_1x1_sbx = self.rbr_conv1x1_sbx_branch.rep_params()
        kernel_1x1_sby, bias_1x1_sby = self.rbr_conv1x1_sby_branch.rep_params()
        kernel_1x1_lpl, bias_1x1_lpl = self.rbr_conv1x1_lpl_branch.rep_params()

        return kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_1x1_3x3_fuse + kernel_identity + kernel_1x1_sbx + kernel_1x1_sby + kernel_1x1_lpl, bias_3x3 + bias_1x1_1x3_3x1_fuse + bias_1x1_sbx + bias_1x1_sby + bias_1x1_lpl

    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias

    def _fuse_1x1_3x3_branch(self, conv1, conv2):
        weight = F.conv2d(conv2.weight.data, conv1.weight.data.permute(1, 0, 2, 3))
        return weight


# 重参数
class RepRFB(nn.Module):
    def __init__(self, feature_nums, act_type='lrelu', deploy=False):
        super(RepRFB, self).__init__()
        self.repblock1 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
        self.repblock2 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)
        self.repblock3 = RepBlock(in_channels=feature_nums, out_channels=feature_nums, act_type=act_type, deploy=deploy)

        self.conv3 = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

        self.esa = ESA(16, feature_nums, nn.Conv2d)
        self.act = activation('lrelu')

    def forward(self, inputs):
        outputs = self.repblock1(inputs)
        outputs = self.repblock2(outputs)
        outputs = self.repblock3(outputs)
        outputs = self.act(self.conv3(outputs))
        outputs = inputs + outputs

        outputs = self.esa(outputs)
        return outputs


class Upsample_Block(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
        super(Upsample_Block, self).__init__()
        self.conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        c = self.conv(input)
        upsample = self.pixel_shuffle(c)
        return upsample


# thanks for RLFN: https://github.com/bytedance/RLFN
class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        #         self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        #         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        #         v_range = self.relu(self.conv_max(v_max))
        #         c3 = self.relu(self.conv3(v_range))
        #         c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RepRFN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_nums=48, upscale_factor=4,
                 deploy=False):
        super(RepRFN, self).__init__()
        self.fea_conv = conv_layer(in_channels=in_channels, out_channels=feature_nums, kernel_size=3)

        self.reprfb1 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        self.reprfb2 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        self.reprfb3 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)
        self.reprfb4 = RepRFB(feature_nums=feature_nums, act_type='lrelu', deploy=deploy)

        self.lr_conv = conv_layer(in_channels=feature_nums, out_channels=feature_nums, kernel_size=3)

        self.upsampler = Upsample_Block(in_channels=feature_nums, out_channels=out_channels,
                                        upscale_factor=upscale_factor)

    def forward(self, inputs):
        outputs_feature = self.fea_conv(inputs)

        outputs = self.reprfb1(outputs_feature)
        outputs = self.reprfb2(outputs)
        outputs = self.reprfb3(outputs)
        outputs = self.reprfb4(outputs)
        outputs = self.lr_conv(outputs)

        outputs = outputs + outputs_feature

        outputs = self.upsampler(outputs)
        return outputs


if __name__ == '__main__':
    from utils.model_summary import get_model_flops, get_model_activation

    model = RepRFN(deploy=True)

    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
