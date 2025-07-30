# -*- coding: utf-8 -*-
# @Time : 2024/3/31 13:10
# @Author : Dengxun
# @Email : 38694034@qq.com
# @File : chconv.py
# @Project : ORGDETv8-main
__all__=['Repchmix','Repchatt']
import math
import numpy as np
import timm.layers
import torch
from timm.layers import trunc_normal_, create_conv2d
from timm.layers.cbam import CbamModule,LightCbamModule
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class ConvNorm(nn.Sequential):
    def __init__(self, in_dim, out_dim, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(in_dim, out_dim, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_dim))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
            device=c.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class NormLinear(nn.Sequential):
    def __init__(self, in_dim, out_dim, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(in_dim))
        self.add_module('l', nn.Linear(in_dim, out_dim, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class ConvMlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, act_layer=nn.GELU):
        super().__init__()
        self.conv1 = ConvNorm(in_dim, hidden_dim, 1, 1, 0)
        self.act = act_layer()
        self.conv2 = ConvNorm(hidden_dim, in_dim, 1, 1, 0, bn_weight_init=0)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))
def val2list(x: list or tuple or any, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1):
    # repeat elements if necessary
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        dropout=0.,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(num_features=out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm_layer=(None, nn.BatchNorm2d),
        act_layer=(None, None),
        kernel_func=nn.ReLU,
        scales=(5,),
        eps=1e-5,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.dim = dim
        self.qkv = ConvNormAct(
            in_channels,
            3 * total_dim,
            1,
            bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    scale,
                    padding=get_same_padding(scale),
                    groups=3 * total_dim,
                    bias=use_bias[0],
                ),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
            )
            for scale in scales
        ])
        self.kernel_func = kernel_func(inplace=False)

        self.proj = ConvNormAct(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def _attn(self, q, k, v):
        dtype = v.dtype
        q, k, v = q.float(), k.float(), v.float()
        kv = k.transpose(-1, -2) @ v
        out = q @ kv
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        return out.to(dtype)

    def forward(self, x):
        B, _, H, W = x.shape

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        multi_scale_qkv = multi_scale_qkv.reshape(B, -1, 3 * self.dim, H * W).transpose(-1, -2)
        q, k, v = multi_scale_qkv.chunk(3, dim=-1)

        # lightweight global attention
        q = F.relu(q)  # self.kernel_func(q)
        k = F.relu(k)  # self.kernel_func(k)
        v = F.pad(v, (0, 1), mode="constant", value=1.)

        if not torch.jit.is_scripting():
            with torch.autocast(device_type=v.device.type, enabled=False):
                out = self._attn(q, k, v)
        else:
            out = self._attn(q, k, v)

        # final projection
        out = out.transpose(-1, -2).reshape(B, -1, H, W)
        out = self.proj(out)
        return out

class ch1dmlp(nn.Module):
    def __init__(self, dim=128,
                 k=3,
                 s=1,
                 ):
        super(ch1dmlp, self).__init__()
        self.ch_conv1d_1 = nn.Sequential(nn.Conv1d(in_channels=dim ,out_channels=dim, kernel_size=9, stride=1, padding='same',
                                     bias=True,groups=dim))
        self.layernorm1=nn.LayerNorm(dim)
        self.act_layer=nn.GELU()
        self.ch_conv1d_2 = nn.Sequential(nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=9, stride=1, padding='same',
                                     bias=True, groups=dim))
        self.layernorm2 = nn.LayerNorm(dim)
        self.mlp=ConvMlp(in_dim=dim,hidden_dim=2*dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        B,C,H,W=x.shape
        x=x.flatten(2)#.transpose(-1,-2)
        x = self.act_layer(self.ch_conv1d_1(x))#row
        x=self.layernorm1(x.transpose(-1,-2)).transpose(-1,-2)
        x=x.reshape(B,C,H,W).permute(0,1,3,2).flatten(2)
        x = self.act_layer(self.ch_conv1d_2(x))#col
        x = self.layernorm2(x.transpose(-1, -2)).transpose(-1, -2)
        x=x.reshape(B,C,W,H).permute(0,1,3,2)
        x=self.mlp(x)
        return x


import torch.nn.functional as F
class ch1d_att(nn.Module):
    def __init__(self, dim=128,
                 k=3,
                 s=1,
                 ):
        super(ch1d_att, self).__init__()
        self.ch_conv1d = nn.Conv1d(in_channels=dim ,out_channels=3*dim, kernel_size=k, stride=s, padding='same',
                                     bias=True,groups=dim)
        self.act_layer=nn.GELU()
        self.mlp=ConvMlp(in_dim=dim,hidden_dim=4*dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        B,C,H,W=x.shape
        x=x.flatten(2)#
        x = self.ch_conv1d(x).transpose(-1,-2)#row
        q,k,v=x[:,:,: C],x[:,:,C: 2*C],x[:,:,2*C: 3*C]
        x=(v*(F.relu(q*k)/F.relu(torch.sum(q*k,dim=1))))#.reshape(B,H,W,C).permute(0,3,1,2)
        out=self.mlp(x)
        return out

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class RepVggDw(nn.Module):
    def __init__(self, ed, kernel_size, legacy=False):
        super().__init__()
        self.conv = ConvNorm(ed, ed, kernel_size, 1, (kernel_size - 1) // 2, groups=ed)
        if legacy:
            self.conv1 = ConvNorm(ed, ed, 1, 1, 0, groups=ed)
            # Make torchscript happy.
            self.bn = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
            self.bn = nn.BatchNorm2d(ed)
        self.dim = ed
        self.legacy = legacy

    def forward(self, x):
        return self.bn(self.conv(x) + self.conv1(x) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()

        if self.legacy:
            conv1 = self.conv1.fuse()
        else:
            conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1, 1, 1, 1]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        if not self.legacy:
            bn = self.bn
            w = bn.weight / (bn.running_var + bn.eps) ** 0.5
            w = conv.weight * w[:, None, None, None]
            b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps) ** 0.5
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
        return conv

class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,  patch_size=3, stride=2, in_chans=3, embed_dim=768,padding=1):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x.reshape(B, H , W , -1).permute(0, 3, 1, 2).contiguous() #x, H, W

class Repchmix(nn.Module):
    def __init__(self, c1=128,
                         c2=128,
                         se=True,

                   ):
        super().__init__()
        self.conv_1 = Conv(c1,c2)
        #self.c2=c1//2
        self.repcov1 = RepConv53(c2//2,c2//2)
        self.repcov2 = RepConv(c2//2, c2//2)
        self.mix=Conv(2*c2,c2,3,1)
        if se:
            self.se = CbamModule(channels=2*c2)  # timm.layers.NonLocalAttn(in_channels=c2)
        else:
            self.se =Conv(c2,c2,k=1)
    def forward(self, x):
        x_ori=x
        x=self.conv_1(x)
        x_left,x_right=x.chunk(2,dim=1)
        x_left = self.repcov1(x_left)
        x_right =self.repcov2(x_right)
        x=torch.cat((x_left,x_right,x),dim=1)
        #print(x.shape)
        x = self.se(x)
        x=self.mix(x)+x_ori
        return x

class Repchatt(nn.Module):
    def __init__(self, c1=128,
                       c2=128,
                        k=3,
                        s=1,
                 ):
        super(Repchatt, self).__init__()
        self.Conv=Conv(c1,c2, k, s)#PatchEmbed(patch_size=2,in_chans=c1,embed_dim=c2,stride=2,padding=0)
        self.se = CbamModule(channels=c2)  # timm.layers.NonLocalAttn(in_channels=c2)
        self.latt=LiteMLA(in_channels=c2,out_channels=c2,dim=16)

    def forward(self, x):
        x=self.Conv(x)
        xrep = self.se(x)
        xrep=self.latt(xrep)+x

        return xrep


import torch
import torch.nn as nn


class RepConv53(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, dilation=1, groups=1,
                 deploy=False):
        super(RepConv53, self).__init__()
        self.deploy = deploy
        padding_3x3 = padding - 1 if kernel_size == 5 else padding

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                                          groups=groups, bias=True)
        else:
            self.conv_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding_3x3,
                                      dilation=dilation, groups=groups, bias=False)
            self.conv_5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(out_channels)
            self.bn_5x5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.deploy:
            return self.reparam_conv(x)
        else:
            out_3x3 = self.bn_3x3(self.conv_3x3(x))
            out_5x5 = self.bn_5x5(self.conv_5x5(x))
            return out_3x3 + out_5x5

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.conv_3x3, self.bn_3x3)
        kernel_5x5, bias_5x5 = self._fuse_bn_tensor(self.conv_5x5, self.bn_5x5)

        reparam_kernel = self._pad_to_kernel_size(kernel_3x3, kernel_5x5.size()) + kernel_5x5
        reparam_bias = bias_3x3 + bias_5x5

        self.reparam_conv = nn.Conv2d(self.conv_5x5.in_channels, self.conv_5x5.out_channels,
                                      kernel_size=self.conv_5x5.kernel_size, stride=self.conv_5x5.stride,
                                      padding=self.conv_5x5.padding, dilation=self.conv_5x5.dilation,
                                      groups=self.conv_5x5.groups, bias=True)
        self.reparam_conv.weight.data = reparam_kernel
        self.reparam_conv.bias.data = reparam_bias

        del self.conv_3x3, self.conv_5x5, self.bn_3x3, self.bn_5x5
        self.deploy = True

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_to_kernel_size(self, kernel, target_size):
        h, w = target_size[-2], target_size[-1]
        padding = (h - kernel.size(2)) // 2
        return nn.functional.pad(kernel, [padding, padding, padding, padding])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

import torch
import torch.nn as nn
import math

class RepConv75(nn.Module):  # Rename the class to reflect the new kernel sizes
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, dilation=1, groups=1,
                 deploy=False):
        super(RepConv75, self).__init__()
        self.deploy = deploy
        padding_5x5 = padding - 1 if kernel_size == 7 else padding

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                                          groups=groups, bias=True)
        else:
            self.conv_5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=padding_5x5,
                                      dilation=dilation, groups=groups, bias=False)
            self.conv_7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn_5x5 = nn.BatchNorm2d(out_channels)
            self.bn_7x7 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.deploy:
            return self.reparam_conv(x)
        else:
            out_5x5 = self.bn_5x5(self.conv_5x5(x))
            out_7x7 = self.bn_7x7(self.conv_7x7(x))
            return out_5x5 + out_7x7

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel_5x5, bias_5x5 = self._fuse_bn_tensor(self.conv_5x5, self.bn_5x5)
        kernel_7x7, bias_7x7 = self._fuse_bn_tensor(self.conv_7x7, self.bn_7x7)

        reparam_kernel = self._pad_to_kernel_size(kernel_5x5, kernel_7x7.size()) + kernel_7x7
        reparam_bias = bias_5x5 + bias_7x7

        self.reparam_conv = nn.Conv2d(self.conv_7x7.in_channels, self.conv_7x7.out_channels,
                                      kernel_size=self.conv_7x7.kernel_size, stride=self.conv_7x7.stride,
                                      padding=self.conv_7x7.padding, dilation=self.conv_7x7.dilation,
                                      groups=self.conv_7x7.groups, bias=True)
        self.reparam_conv.weight.data = reparam_kernel
        self.reparam_conv.bias.data = reparam_bias

        del self.conv_5x5, self.conv_7x7, self.bn_5x5, self.bn_7x7
        self.deploy = True

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_to_kernel_size(self, kernel, target_size):
        h, w = target_size[-2], target_size[-1]
        padding = (h - kernel.size(2)) // 2
        return nn.functional.pad(kernel, [padding, padding, padding, padding])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


if __name__=="__main__":
    data = torch.ones(1, 128, 40,40)
    model=RepConv53(128,128)
    print(model(data).shape)