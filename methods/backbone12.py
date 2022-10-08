# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import numpy as np
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
#from rfconv import RFConv2d
# --- gaussian initialize ---#
def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)#torch.norm默认求2范数，求第二维2范数
        x_normalized = x.div(x_norm + 0.00001)#x/x_norm+0.0001
        L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = 10 * cos_dist
        return scores


# --- flatten tensor ---
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# --- LSTMCell module for matchingnet ---
class LSTMCell(nn.Module):
    maml = True

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if self.maml:
            self.x2h = Linear_fw(input_size, 4 * hidden_size, bias=bias)
            self.h2h = Linear_fw(hidden_size, 4 * hidden_size, bias=bias)
        else:
            self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
            self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        if hidden is None:
            hx = torch.zeors_like(x)
            cx = torch.zeros_like(x)
        else:
            hx, cx = hidden

        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)


# --- LSTM module for matchingnet 递归神经网络---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        assert (self.num_layers == 1)

        self.lstm = LSTMCell(input_size, hidden_size, self.bias)

    def forward(self, x, hidden=None):
        # swap axis if batch first
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h0, c0 = hidden

        # forward
        outs = []
        hn = h0[0]
        cn = c0[0]
        for seq in range(x.size(0)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn.unsqueeze(0))
        outs = torch.cat(outs, dim=0)

        # reverse foward
        if self.num_directions == 2:
            outs_reverse = []
            hn = h0[1]
            cn = c0[1]
            for seq in range(x.size(0)):
                seq = x.size(1) - 1 - seq
                hn, cn = self.lstm(x[seq], (hn, cn))
                outs_reverse.append(hn.unsqueeze(0))
            outs_reverse = torch.cat(outs_reverse, dim=0)
            outs = torch.cat([outs, outs_reverse], dim=2)

        # swap axis if batch first
        if self.batch_first:
            outs = outs.permute(1, 0, 2)
        return outs


# --- Linear module ---
class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_fw, self).__init__(in_features, out_features, bias=bias)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


# --- softplus module ---
def softplus(x):
    return torch.nn.functional.softplus(x, beta=100)

class GaussianBluerConv2(nn.Module):
    def __init__(self, out_features):
        super(GaussianBluerConv2, self).__init__()
        self.out_features = out_features
        self.filters = torch.nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        filters = self.filters.unsqueeze(0).unsqueeze(0)
        filters = filters.repeat(self.out_features, self.out_features, 1, 1)
        x1 = F.conv2d(x, filters, stride=1, padding=1)
        x2 = x-x1
        return x2

# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = True

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(
                self.beta)).expand_as(out)
            out = gamma * out + beta
        return out




# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out


# --- Simple Conv Block ---
class ConvBlock(nn.Module):
    maml = True

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = FeatureWiseTransformation2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out



class SimpleBlock(nn.Module):
    maml = True

    def __init__(self, indim, outdim, half_res, leaky=True):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            # self.gaosi = GaussianBluerConv2(indim)
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1,padding=1, bias=False)           
            self.BN1 = BatchNorm2d_fw(outdim)
            if self.outdim!=64:
                self.C3 = Conv2d_fw(outdim, outdim, kernel_size=3, stride=1, bias=False, padding=1)
                self.BN3 = FeatureWiseTransformation2d_fw(outdim)
            # self.C2 = SplAtConv2d(outdim, outdim, kernel_size=3, stride=1, padding=1, bias=False)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, stride=1, bias=False, padding=1)
            self.BN2 = FeatureWiseTransformation2d_fw(outdim)

        else:
            # self.gaosi = GaussianBluerConv2(indim)
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)#第一层的图片大小不变，后面三层的变为二分之一
            self.BN1 = nn.BatchNorm2d(outdim)
            # self.C2 = SplAtConv2d(outdim, outdim, kernel_size=3, stride=1, padding=1, bias=False)#图片大小不变
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, stride=1, bias=False, padding=1)
            self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu3 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        self.relu2 = nn.ReLU(inplace=True) if not leaky else nn.LeakyReLU(0.2, inplace=True)
        if self.outdim != 64:
            self.parametrized_layers = [self.C1, self.C3, self.C2, self.BN1, self.BN3, self.BN2]
            
        else:
            self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2, bias=False)
                self.BNshortcut = FeatureWiseTransformation2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        if self.outdim != 64:
            out = self.C3(out)
            out = self.BN3(out)
        
        out3 = self.C2(out)
        out = self.BN2(out3)

        if self.shortcut_type == 'identity':
            short_out = x
        else:
            short_out = self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


class PSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, parts=4, bias=False):
        super(PSConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // parts
        _out_channels = out_channels // parts
        for i in range(parts):
            self.mask[i * _out_channels: (i + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
            self.mask[(i + parts//2)%parts * _out_channels: ((i + parts//2)%parts + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)#分割，列分割
        x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
        return self.gwconv(x) + self.conv(x) + x_shift



class PSGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, parts=4, bias=False):
        super(PSGConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=groups * parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=groups * parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // (groups * parts)
        _out_channels = out_channels // (groups * parts)
        for i in range(parts):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
                self.mask[((i + parts // 2) % parts + j * groups) * _out_channels: ((i + parts // 2) % parts + j * groups + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.gwconv_shift(x_merge)
        return self.gwconv(x) + self.conv(x) + x_shift




# --- ConvNet module ---
class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


# --- ConvNetNopool module ---
class ConvNetNopool(nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        self.grads = []
        self.fmaps = []
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


# --- ResNet module ---
class ResNet(nn.Module):
    maml = True

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True, leakyrelu=True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        self.grads = []
        self.fmaps = []
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #my_conv1 = SplAtConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            #my_conv1 = SplAtConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn1 = nn.BatchNorm2d(64)
            #16*64*112*112
        relu1 = nn.ReLU(inplace=True) if not leakyrelu else nn.LeakyReLU(0.2, inplace=True) #inplace=true对上面传下来的值进行覆盖
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #16*64*56*56


        init_layer(conv1)
        init_layer(bn1)

        # trunk = [conv1, bn1, relu1, conv_lg, pool1]
        trunk = [conv1, bn1, pool1, relu1]

        indim = 64
        for i in range(4): #resnet网络
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0) #half_res = 0
                B = block(indim, list_of_out_dims[i], half_res, leaky=leakyrelu)
                trunk.append(B)
                indim = list_of_out_dims[i]


        if flatten:
            avgpool = nn.AvgPool2d(7)#全局平均值代替全连接
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]


        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out

#from dropblock import DropBlock2D

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=True, rectify_avg=True, norm_layer=None,
                 dropblock_prob=0.3, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)#32
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.conv1 = Conv2d(in_channels,1,kernel_size=1,bias=False)
        self.norm = nn.Sigmoid()
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):

        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]#16*channel
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
            # out1 = []
            # for (att,split) in zip(attens,splited):
            #     spation = self.conv1(split)
            #     spation = self.norm(spation)
            #     out = split*att+spation*split
            #     out1.append(out)
            # out = sum(out1)
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class CropLayer(nn.Module):
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()

        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop,self.cols_to_crop: -self.cols_to_crop]

class CA_Block(nn.Module):
  def __init__(self, channel, h, w, reduction=16):
    super(CA_Block, self).__init__()

    self.h = h
    self.w = w

    self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
    self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

    self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                              bias=False)

    self.relu = nn.ReLU()
    self.bn = nn.BatchNorm2d(channel // reduction)

    self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
    self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

    self.sigmoid_h = nn.Sigmoid()
    self.sigmoid_w = nn.Sigmoid()

  def forward(self, x):
    x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
    x_w = self.avg_pool_y(x)

    x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

    x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

    s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
    s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

    out = x * s_h.expand_as(x) * s_w.expand_as(x)

    return out
# --- Conv networks ---
def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


# --- ResNet networks ---
# def ResNet10(flatten=True, leakyrelu=True):
def ResNet10(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten, leakyrelu)#leakyrelu


def ResNet18(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten, leakyrelu)


def ResNet34(flatten=True, leakyrelu=False):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten, leakyrelu)


model_dict = dict(Conv4=Conv4,
                  Conv6=Conv6,
                  ResNet10=ResNet10,
                  ResNet18=ResNet18,
                  ResNet34=ResNet34)
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

