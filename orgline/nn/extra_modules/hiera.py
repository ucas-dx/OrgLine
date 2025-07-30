import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init

from sam2.build_sam import build_sam2
__all__=['hiera','hiera2',"hiera3",'samhierablock',"hiera3_tiny"]
import timm
from orgline.nn.modules import TransformerBlock

import warnings
import math
from functools import partial
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, create_attn, get_attn, \
    get_act_layer, get_norm_layer, create_classifier
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations
warnings.filterwarnings("ignore", message="Overwriting .* in registry"
)
warnings.filterwarnings("ignore", message="Argument(s) .* are not valid for transform"
)
warnings.filterwarnings("ignore", message="ShiftScaleRotate is a special case of Affine transform"
)
warnings.filterwarnings("ignore", category=FutureWarning)
class SEACttention(nn.Module):
    def __init__(self, channel=512,outchannel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.Dw=nn.Sequential(Conv(channel,channel),Conv(channel,outchannel))
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
        outy=x * y.expand_as(x)
        out=self.Dw(outy)
        return out



def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=64,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        first_dilation=None,
        norm_layer=None,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob=0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn,
        channels,
        block_repeats,
        inplanes,
        reduce_first=1,
        output_stride=32,
        down_kernel_size=1,
        avg_down=False,
        drop_block_rate=0.,
        drop_path_rate=0.,
        **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info




class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            stem_width=64,
            stem_type='',
            replace_stem_pool=False,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            aa_layer=None,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem pooling. The name 'maxpool' remains for weight compatibility.
        if replace_stem_pool:
            self.maxpool = nn.Sequential(*filter(None, [
                nn.Conv2d(inplanes, inplanes, 3, stride=1 if aa_layer else 2, padding=1, bias=False),
                create_aa(aa_layer, channels=inplanes, stride=2) if aa_layer is not None else None,
                norm_layer(inplanes),
                act_layer(inplace=True),
            ]))
        else:
            if aa_layer is not None:
                if issubclass(aa_layer, nn.AvgPool2d):
                    self.maxpool = aa_layer(2)
                else:
                    self.maxpool = nn.Sequential(*[
                        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                        aa_layer(channels=inplanes, stride=2)])
            else:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only=False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        features=[]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            features.append(x)
            x = self.layer2(x)
            features.append(x)
            x = self.layer3(x)
            features.append(x)
            x = self.layer4(x)
            features.append(x)
        return features

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.forward_head(x)
        return x



def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


def _tcfg(url='', **kwargs):
    return _cfg(url=url, **dict({'interpolation': 'bicubic'}, **kwargs))


def _ttcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'test_input_size': (3, 288, 288), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models',
    }, **kwargs))


def _rcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'crop_pct': 0.95, 'test_input_size': (3, 288, 288), 'test_crop_pct': 1.0,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476'
    }, **kwargs))


def _r3cfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic', 'input_size': (3, 160, 160), 'pool_size': (5, 5),
        'crop_pct': 0.95, 'test_input_size': (3, 224, 224), 'test_crop_pct': 0.95,
        'origin_url': 'https://github.com/huggingface/pytorch-image-models', 'paper_ids': 'arXiv:2110.00476',
    }, **kwargs))


def _gcfg(url='', **kwargs):
    return _cfg(url=url, **dict({
        'interpolation': 'bicubic',
        'origin_url': 'https://cv.gluon.ai/model_zoo/classification.html',
    }, **kwargs))


default_cfgs = generate_default_cfgs({
    # ResNet and Wide ResNet trained w/ timm (RSB paper and others)
    'resnet10t.c3_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet10t_176_c3-f3215ab1.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'resnet14t.c3_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet14t_176_c3-c4ed2c37.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_crop_pct=0.95, test_input_size=(3, 224, 224),
        first_conv='conv1.0'),
    'resnet18.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth'),
    'resnet18.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth'),
    'resnet18.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a3_0-40c531c8.pth'),
    'resnet18d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth',
        first_conv='conv1.0'),
    'resnet34.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth'),
    'resnet34.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth'),
    'resnet34.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth',
        crop_pct=0.95),
    'resnet34.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth'),
    'resnet34d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34d_ra2-f8dcfcaf.pth',
        first_conv='conv1.0'),
    'resnet26.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26-9aa10e23.pth'),
    'resnet26d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet26d-69e92c46.pth',
        first_conv='conv1.0'),
    'resnet26t.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-attn-weights/resnet26t_256_ra2-6f6fa748.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.94, test_input_size=(3, 320, 320), test_crop_pct=1.0),
    'resnet50.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth'),
    'resnet50.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1h2_176-001a1197.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), crop_pct=0.9, test_input_size=(3, 224, 224), test_crop_pct=1.0),
    'resnet50.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a2_0-a2746f79.pth'),
    'resnet50.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a3_0-59cae1ef.pth'),
    'resnet50.b1k_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b1k-532a802a.pth'),
    'resnet50.b2k_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_b2k-1ba180c1.pth'),
    'resnet50.c1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c1-5ba5e060.pth'),
    'resnet50.c2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_c2-d01e05b2.pth'),
    'resnet50.d_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_d-f39db8af.pth'),
    'resnet50.ram_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth'),
    'resnet50.am_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_am-6c502b37.pth'),
    'resnet50.ra_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnet50_ra-85ebb6e5.pth'),
    'resnet50.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/rw_resnet50-86acaeed.pth'),
    'resnet50d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50d_ra2-464e36ba.pth',
        first_conv='conv1.0'),
    'resnet50d.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth',
        first_conv='conv1.0'),
    'resnet50d.a2_in1k': _rcfg(
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a2_0-a3adc64d.pth',
        first_conv='conv1.0'),
    'resnet50d.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a3_0-403fdfad.pth',
        first_conv='conv1.0'),
    'resnet50t.untrained': _ttcfg(first_conv='conv1.0'),
    'resnet101.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth'),
    'resnet101.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth'),
    'resnet101.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a2_0-6edb36c7.pth'),
    'resnet101.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a3_0-1db14157.pth'),
    'resnet101d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet152.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1h-dc400468.pth'),
    'resnet152.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1_0-2eee8a7a.pth'),
    'resnet152.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a2_0-b4c6978f.pth'),
    'resnet152.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a3_0-134d4688.pth'),
    'resnet152d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet152d_ra2-5cac0439.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'resnet200.untrained': _ttcfg(),
    'resnet200d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)),
    'wide_resnet50_2.racm_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth'),

    # torchvision resnet weights
    'resnet18.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet18-5c106cde.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet34.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-19c8e357.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet50.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet101.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet101-cd907fc2.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnet152.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnet152-f82ba261.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet50_2.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet50_2.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet101_2.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'wide_resnet101_2.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

    # ResNets w/ alternative norm layers
    'resnet50_gn.a1h_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth',
        crop_pct=0.94),

    # ResNeXt trained in timm (RSB paper and others)
    'resnext50_32x4d.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1h-0146ab0a.pth'),
    'resnext50_32x4d.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a1_0-b5a91a1d.pth'),
    'resnext50_32x4d.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a2_0-efc76add.pth'),
    'resnext50_32x4d.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnext50_32x4d_a3_0-3e450271.pth'),
    'resnext50_32x4d.ra_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/resnext50_32x4d_ra-d733960d.pth'),
    'resnext50d_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnext50d_32x4d-103e99f8.pth',
        first_conv='conv1.0'),
    'resnext101_32x4d.untrained': _ttcfg(),
    'resnext101_64x4d.c1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnext101_64x4d_c-0d0e0cc0.pth'),

    # torchvision ResNeXt weights
    'resnext50_32x4d.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext101_32x8d.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext101_64x4d.tv_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth',
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext50_32x4d.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),
    'resnext101_32x8d.tv2_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth',
        input_size=(3, 176, 176), pool_size=(6, 6), test_input_size=(3, 224, 224), test_crop_pct=0.965,
        license='bsd-3-clause', origin_url='https://github.com/pytorch/vision'),

    #  ResNeXt models - Weakly Supervised Pretraining on Instagram Hashtags
    #  from https://github.com/facebookresearch/WSL-Images
    #  Please note the CC-BY-NC 4.0 license on these weights, non-commercial use only.
    'resnext101_32x8d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'resnext101_32x16d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'resnext101_32x32d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),
    'resnext101_32x48d.fb_wsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/WSL-Images'),

    #  Semi-Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'resnet18.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnet50.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),

    #  Semi-Weakly Supervised ResNe*t models from https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
    #  Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only.
    'resnet18.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnet50.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext50_32x4d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x4d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x8d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),
    'resnext101_32x16d.fb_swsl_ig1b_ft_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth',
        license='cc-by-nc-4.0', origin_url='https://github.com/facebookresearch/semi-supervised-ImageNet1K-models'),

    #  Efficient Channel Attention ResNets
    'ecaresnet26t.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet26t_ra2-46609757.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        test_crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnetlight.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnetlight-75a9c627.pth',
        test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet50d.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d-93c81e3b.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet50d_pruned.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet50d_p-e4fa23c2.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet50t.ra2_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet50t_ra2-f7ac63c4.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8),
        test_crop_pct=0.95, test_input_size=(3, 320, 320)),
    'ecaresnet50t.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a1_0-99bd76a8.pth',
        first_conv='conv1.0'),
    'ecaresnet50t.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a2_0-b1c7b745.pth',
        first_conv='conv1.0'),
    'ecaresnet50t.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/ecaresnet50t_a3_0-8cc311f1.pth',
        first_conv='conv1.0'),
    'ecaresnet101d.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d-153dad65.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet101d_pruned.miil_in1k': _tcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/ecaresnet101d_p-9e74cb91.pth',
        first_conv='conv1.0', test_crop_pct=0.95, test_input_size=(3, 288, 288)),
    'ecaresnet200d.untrained': _ttcfg(
        first_conv='conv1.0', input_size=(3, 256, 256), crop_pct=0.95, pool_size=(8, 8)),
    'ecaresnet269d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/ecaresnet269d_320_ra2-7baa55cb.pth',
        first_conv='conv1.0', input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 352, 352)),

    #  Efficient Channel Attention ResNeXts
    'ecaresnext26t_32x4d.untrained': _tcfg(first_conv='conv1.0'),
    'ecaresnext50t_32x4d.untrained': _tcfg(first_conv='conv1.0'),

    #  Squeeze-Excitation ResNets, to eventually replace the models in senet.py
    'seresnet18.untrained': _ttcfg(),
    'seresnet34.untrained': _ttcfg(),
    'seresnet50.a1_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a1_0-ffa00869.pth',
        crop_pct=0.95),
    'seresnet50.a2_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a2_0-850de0d9.pth',
        crop_pct=0.95),
    'seresnet50.a3_in1k': _r3cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/seresnet50_a3_0-317ecd56.pth',
        crop_pct=0.95),
    'seresnet50.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet50_ra_224-8efdb4bb.pth'),
    'seresnet50t.untrained': _ttcfg(
        first_conv='conv1.0'),
    'seresnet101.untrained': _ttcfg(),
    'seresnet152.untrained': _ttcfg(),
    'seresnet152d.ra2_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet152d_ra2-04464dd2.pth',
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_crop_pct=1.0, test_input_size=(3, 320, 320)
    ),
    'seresnet200d.untrained': _ttcfg(
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8)),
    'seresnet269d.untrained': _ttcfg(
        first_conv='conv1.0', input_size=(3, 256, 256), pool_size=(8, 8)),

    #  Squeeze-Excitation ResNeXts, to eventually replace the models in senet.py
    'seresnext26d_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26d_32x4d-80fa48a3.pth',
        first_conv='conv1.0'),
    'seresnext26t_32x4d.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26tn_32x4d-569cb627.pth',
        first_conv='conv1.0'),
    'seresnext50_32x4d.racm_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext50_32x4d_racm-a304a460.pth'),
    'seresnext101_32x4d.untrained': _ttcfg(),
    'seresnext101_32x8d.ah_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101_32x8d_ah-e6bc4c0a.pth'),
    'seresnext101d_32x8d.ah_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnext101d_32x8d_ah-191d7b94.pth',
        first_conv='conv1.0'),

    # ResNets with anti-aliasing / blur pool
    'resnetaa50d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/',
        first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'resnetaa101d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/',
        first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'seresnextaa101d_32x8d.sw_in12k_ft_in1k_288': _ttcfg(
        hf_hub_id='timm/',
        crop_pct=0.95, input_size=(3, 288, 288), pool_size=(9, 9), test_input_size=(3, 320, 320), test_crop_pct=1.0,
        first_conv='conv1.0'),
    'seresnextaa101d_32x8d.sw_in12k_ft_in1k': _ttcfg(
        hf_hub_id='timm/',
        first_conv='conv1.0', test_crop_pct=1.0),

    'resnetaa50d.sw_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'resnetaa50d.d_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'resnetaa101d.sw_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),
    'seresnextaa101d_32x8d.sw_in12k': _ttcfg(
        hf_hub_id='timm/',
        num_classes=11821, first_conv='conv1.0', crop_pct=0.95, test_crop_pct=1.0),

    'resnetblur18.untrained': _ttcfg(),
    'resnetblur50.bt_in1k': _ttcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnetblur50-84f4748f.pth'),
    'resnetblur50d.untrained': _ttcfg(first_conv='conv1.0'),
    'resnetblur101d.untrained': _ttcfg(first_conv='conv1.0'),
    'resnetaa50.a1h_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnetaa50_a1h-4cf422b3.pth'),

    'seresnetaa50d.untrained': _ttcfg(first_conv='conv1.0'),
    'seresnextaa101d_32x8d.ah_in1k': _rcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/seresnextaa101d_32x8d_ah-83c8ae12.pth',
        first_conv='conv1.0'),

    # ResNet-RS models
    'resnetrs50.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs50_ema-6b53758b.pth',
        input_size=(3, 160, 160), pool_size=(5, 5), crop_pct=0.91, test_input_size=(3, 224, 224),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs101.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs101_i192_ema-1509bbf6.pth',
        input_size=(3, 192, 192), pool_size=(6, 6), crop_pct=0.94, test_input_size=(3, 288, 288),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs152.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs152_i256_ema-a9aff7f9.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs200.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tpu-weights/resnetrs200_c-6b698b88.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 320, 320),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs270.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs270_ema-b40e674c.pth',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0, test_input_size=(3, 352, 352),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs350.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs350_i256_ema-5a1aa8f1.pth',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0, test_input_size=(3, 384, 384),
        interpolation='bicubic', first_conv='conv1.0'),
    'resnetrs420.tf_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rs-weights/resnetrs420_ema-972dee69.pth',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0, test_input_size=(3, 416, 416),
        interpolation='bicubic', first_conv='conv1.0'),

    # gluon resnet weights
    'resnet18.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet18_v1b-0757602b.pth'),
    'resnet34.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth'),
    'resnet50.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1b-0ebe02e2.pth'),
    'resnet101.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1b-3b017079.pth'),
    'resnet152.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1b-c1edb0dd.pth'),
    'resnet50c.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1c-48092f55.pth',
        first_conv='conv1.0'),
    'resnet101c.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1c-1f26822a.pth',
        first_conv='conv1.0'),
    'resnet152c.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1c-a3bb0b98.pth',
        first_conv='conv1.0'),
    'resnet50d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1d-818a1b1b.pth',
        first_conv='conv1.0'),
    'resnet101d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1d-0f9c8644.pth',
        first_conv='conv1.0'),
    'resnet152d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1d-bd354e12.pth',
        first_conv='conv1.0'),
    'resnet50s.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet50_v1s-1762acc0.pth',
        first_conv='conv1.0'),
    'resnet101s.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet101_v1s-60fe0cc1.pth',
        first_conv='conv1.0'),
    'resnet152s.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet152_v1s-dcc41b81.pth',
        first_conv='conv1.0'),
    'resnext50_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext50_32x4d-e6a097c1.pth'),
    'resnext101_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_32x4d-b253c8c4.pth'),
    'resnext101_64x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnext101_64x4d-f9a8e184.pth'),
    'seresnext50_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext50_32x4d-90cf2d6e.pth'),
    'seresnext101_32x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_32x4d-cf52900d.pth'),
    'seresnext101_64x4d.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_seresnext101_64x4d-f9926f93.pth'),
    'senet154.gluon_in1k': _gcfg(
        hf_hub_id='timm/',
        url='https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_senet154-70a1a3c0.pth',
        first_conv='conv1.0'),
})


@register_model
def resnet10t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-10-T model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet10t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet14t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-14-T model.
    """
    model_args = dict(block=Bottleneck, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet14t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet('resnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet18d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet18d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet('resnet34', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet34d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet34d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return _create_resnet('resnet26', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-T model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet26t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet26d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet26d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50c(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet50c', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet50d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet50s', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50t(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-T model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet('resnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101c(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet101c', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet101d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet101s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet101s', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet('resnet152', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152c(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet152c', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet152d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet152s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet152s', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet('resnet200', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet200d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet50_2(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return _create_resnet('wide_resnet50_2', pretrained, **dict(model_args, **kwargs))


@register_model
def wide_resnet101_2(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128)
    return _create_resnet('wide_resnet101_2', pretrained, **dict(model_args, **kwargs))


@register_model
def resnet50_gn(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model w/ GroupNorm
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet50_gn', pretrained, norm_layer=GroupNorm, **model_args)


@register_model
def resnext50_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4)
    return _create_resnet('resnext50_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext50d_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnext50d_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4)
    return _create_resnet('resnext101_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x8d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8)
    return _create_resnet('resnext101_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x16d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x16d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16)
    return _create_resnet('resnext101_32x16d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_32x32d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x32d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=32)
    return _create_resnet('resnext101_32x32d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnext101_64x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4)
    return _create_resnet('resnext101_64x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet26t(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet26t', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet50d_pruned(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model pruned with eca.
        The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50d_pruned', pretrained, pruned=True, **dict(model_args, **kwargs))


@register_model
def ecaresnet50t(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnetlight(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D light model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[1, 1, 11, 3], stem_width=32, avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnetlight', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet101d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet101d_pruned(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model pruned with eca.
       The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet101d_pruned', pretrained, pruned=True, **dict(model_args, **kwargs))


@register_model
def ecaresnet200d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnet269d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet269d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnext26t_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnext26t_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def ecaresnext50t_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-50-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnext50t_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet18(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet18', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet34(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet34', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet50', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet50t(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep_tiered',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet50t', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet101(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet101', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet152', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet152d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet152d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet200d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet200d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnet269d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet269d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext26d_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-D model.`
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
    combination of deep stem and avg_pool in downsample.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext26d_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext26t_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext26t_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext26tn_32x4d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-T model.
    NOTE I deprecated previous 't' model defs and replaced 't' with 'tn', this was the only tn model of note
    so keeping this def for backwards compat with any uses out there. Old 't' model is lost.
    """
    return seresnext26t_32x4d(pretrained=pretrained, **kwargs)


@register_model
def seresnext50_32x4d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext50_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_32x4d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_32x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_32x8d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101d_32x8d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101d_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnext101_64x4d(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_64x4d', pretrained, **dict(model_args, **kwargs))


@register_model
def senet154(pretrained=False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4, stem_type='deep',
        down_kernel_size=3, block_reduce_first=2, block_args=dict(attn_layer='se'))
    return _create_resnet('senet154', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur18(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model with blur anti-aliasing
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d)
    return _create_resnet('resnetblur18', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with blur anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d)
    return _create_resnet('resnetblur50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetblur50d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetblur101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetblur101d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa34d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model w/ avgpool anti-aliasing
    """
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d, stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa34d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with avgpool anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d)
    return _create_resnet('resnetaa50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa50d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetaa101d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa101d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnetaa50d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE=ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnetaa50d', pretrained, **dict(model_args, **kwargs))


@register_model
def seresnextaa101d_32x8d(pretrained=False, **kwargs) -> ResNet:
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        stem_width=32, stem_type='deep', avg_down=True, aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnextaa101d_32x8d', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs50(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs50', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs101(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs101', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs152(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs152', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs200(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs200', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs270(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 29, 53, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs270', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs350(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 36, 72, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs350', pretrained, **dict(model_args, **kwargs))


@register_model
def resnetrs420(pretrained=False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 44, 87, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True, block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs420', pretrained, **dict(model_args, **kwargs))


register_model_deprecations(__name__, {
    'tv_resnet34': 'resnet34.tv_in1k',
    'tv_resnet50': 'resnet50.tv_in1k',
    'tv_resnet101': 'resnet101.tv_in1k',
    'tv_resnet152': 'resnet152.tv_in1k',
    'tv_resnext50_32x4d': 'resnext50_32x4d.tv_in1k',
    'ig_resnext101_32x8d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ig_resnext101_32x16d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ig_resnext101_32x32d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ig_resnext101_32x48d': 'resnext101_32x8d.fb_wsl_ig1b_ft_in1k',
    'ssl_resnet18': 'resnet18.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnet50': 'resnet50.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext50_32x4d': 'resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext101_32x4d': 'resnext101_32x4d.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext101_32x8d': 'resnext101_32x8d.fb_ssl_yfcc100m_ft_in1k',
    'ssl_resnext101_32x16d': 'resnext101_32x16d.fb_ssl_yfcc100m_ft_in1k',
    'swsl_resnet18': 'resnet18.fb_swsl_ig1b_ft_in1k',
    'swsl_resnet50': 'resnet50.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext50_32x4d': 'resnext50_32x4d.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext101_32x4d': 'resnext101_32x4d.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext101_32x8d': 'resnext101_32x8d.fb_swsl_ig1b_ft_in1k',
    'swsl_resnext101_32x16d': 'resnext101_32x16d.fb_swsl_ig1b_ft_in1k',
    'gluon_resnet18_v1b': 'resnet18.gluon_in1k',
    'gluon_resnet34_v1b': 'resnet34.gluon_in1k',
    'gluon_resnet50_v1b': 'resnet50.gluon_in1k',
    'gluon_resnet101_v1b': 'resnet101.gluon_in1k',
    'gluon_resnet152_v1b': 'resnet152.gluon_in1k',
    'gluon_resnet50_v1c': 'resnet50c.gluon_in1k',
    'gluon_resnet101_v1c': 'resnet101c.gluon_in1k',
    'gluon_resnet152_v1c': 'resnet152c.gluon_in1k',
    'gluon_resnet50_v1d': 'resnet50d.gluon_in1k',
    'gluon_resnet101_v1d': 'resnet101d.gluon_in1k',
    'gluon_resnet152_v1d': 'resnet152d.gluon_in1k',
    'gluon_resnet50_v1s': 'resnet50s.gluon_in1k',
    'gluon_resnet101_v1s': 'resnet101s.gluon_in1k',
    'gluon_resnet152_v1s': 'resnet152s.gluon_in1k',
    'gluon_resnext50_32x4d': 'resnext50_32x4d.gluon_in1k',
    'gluon_resnext101_32x4d': 'resnext101_32x4d.gluon_in1k',
    'gluon_resnext101_64x4d': 'resnext101_64x4d.gluon_in1k',
    'gluon_seresnext50_32x4d': 'seresnext50_32x4d.gluon_in1k',
    'gluon_seresnext101_32x4d': 'seresnext101_32x4d.gluon_in1k',
    'gluon_seresnext101_64x4d': 'seresnext101_64x4d.gluon_in1k',
    'gluon_senet154': 'senet154.gluon_in1k',
})

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



class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


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
            if not hasattr(self, "id_tensor"):
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
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the ORGDETv8 mask Proto module."""
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """ORGDETv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the ORGDETv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for ORGDETv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the ORGDET FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for ORGDET-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for ORGDET-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p





class samhiera(nn.Module):
    def __init__(self,
                 num_class=1,
                 layerchan=[32, 64, 128, 256],
                 depths=[2,2,6,2]):
        super(samhiera,self).__init__()
        checkpoint = "sam2_hiera_large.pt"
        model_cfg = r"sam2_hiera_l.yaml"
        self.convsteam=nn.Sequential(Conv(3, layerchan[0]//2, 3, 2, 1),
                                     C2f(layerchan[0]//2, layerchan[0]//2,shortcut=True),
                                     Conv(layerchan[0]//2, layerchan[0], 3, 2, 1)
                                     )
        self.model = build_sam2(model_cfg, checkpoint, device="cpu", ).image_encoder
        for param in self.model.parameters():
            param.requires_grad = False
        # for param in self.model.neck.parameters():
        #     param.requires_grad = True
        self.downconv = nn.Sequential(Conv(256,256,3,2),ResNetLayer(256,256))
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
    def forward(self, x):
        # print(seg_size)
        """
        features = [torch.ones([4, 128, 128, 128]),
                    torch.ones([4, 256, 64, 64]),
                    torch.ones([4, 512, 32, 32]),
                    torch.ones([4, 1024, 16, 16])]
        """
        #
        x0 = self.convsteam(x)
        features =  self.model(x)['backbone_fpn']
        feature3 = self.downconv(features[-1])
        features.append(feature3)
        #print(features)
        features.insert(0, x0)
        return features

class samhiera2(nn.Module):
    def __init__(self,
                 num_class=1,
                 layerchan=[32, 64, 128, 256],
                 depths=[2,2,6,2]):
        super(samhiera2,self).__init__()
        checkpoint = "sam2_hiera_large.pt"
        model_cfg = r"sam2_hiera_l.yaml"
        self.model = build_sam2(model_cfg, checkpoint, device="cpu", ).image_encoder
        #self.model.neck=nn.Identity()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.neck.parameters():
            param.requires_grad = True
        self.downconv = nn.Sequential(Conv(256, 256, 3, 2), ResNetLayer(256, 256))
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        # print(seg_size)
        """
        features = [torch.ones([4, 128, 128, 128]),
                    torch.ones([4, 256, 64, 64]),
                    torch.ones([4, 512, 32, 32]),
                    torch.ones([4, 1024, 16, 16])]
        """
        #
        features = self.model(x)['backbone_fpn']
        feature3 = self.downconv(features[-1])
        features.append(feature3)
        # print(features)
        return features


class samhiera3(nn.Module):
    def __init__(self,
                 num_class=1,
                 layerchan1=[64, 128, 256, 512],
                 layerchan2=[144, 288, 576, 1152],
                 layerchan3=[128, 256, 512, 1024],
                 depths=[2, 2, 2, 2]):
        super(samhiera3, self).__init__()
        model_cfg = r"sam2_hiera_l.yaml"
        # Build the model (only the image encoder trunk is used)
        self.model = build_sam2(
            model_cfg,
            None,
            device="cpu",
            config_dir=r'orgline/sam2_configs'
        ).image_encoder.trunk
        # ResNet backbone
        self.modelres = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
        # Freeze SAM2 parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # SEACttention modules for channel adjustment and attention
        self.channel_convs = nn.ModuleList([
            SEACttention(
                channel=layerchan1[i] + layerchan2[i],
                outchannel=layerchan3[i],
            )
            for i in range(len(layerchan1))
        ])
        # Store output channel sizes for each feature level
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        out_features = []
        # Get features from both backbones
        features1 = self.model(x)
        features2 = self.modelres(x)
        # Concatenate and adjust channels for each feature level
        for i in range(len(features1)):
            # Concatenate features1[i] and features2[i] along the channel dimension
            combined_features = torch.cat((features1[i], features2[i]), dim=1)
            # Adjust the number of channels to layerchan3[i] using SEACttention
            adjusted_features = self.channel_convs[i](combined_features)
            out_features.append(adjusted_features)
        return out_features



class samhiera3_tiny(nn.Module):
    def __init__(self,
                 num_class=1,
                 layerchan1=[64, 128, 256, 512],
                 layerchan2=[96, 192, 384, 768],
                 layerchan3=[64, 128, 256, 512],
                 depths=[2, 2, 2, 2]):
        super(samhiera3_tiny, self).__init__()
        model_cfg = r"sam2_hiera_t.yaml"
        checkpoint = "sam2_hiera_tiny.pt"
        # Build the model (only the image encoder trunk is used)
        self.model = build_sam2(
            model_cfg,
            checkpoint,
            device="cpu",
            config_dir=r'orgline/sam2_configs'
        ).image_encoder.trunk
        # ResNet backbone
        self.modelres = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

        # Freeze the SAM2 model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # SEACttention modules for channel adjustment and attention
        self.channel_convs = nn.ModuleList([
            SEACttention(
                channel=layerchan1[i] + layerchan2[i],
                outchannel=layerchan3[i],
            )
            for i in range(len(layerchan1))
        ])
        # Store output channel sizes for each feature level
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]

    def forward(self, x):
        out_features = []
        # Get features from both backbones
        features1 = self.model(x)
        features2 = self.modelres(x)
        # Concatenate and adjust channels for each feature level
        for i in range(len(features1)):
            # Concatenate features1[i] and features2[i] along the channel dimension
            combined_features = torch.cat((features1[i], features2[i]), dim=1)
            # Adjust the number of channels to layerchan3[i] using SEACttention
            adjusted_features = self.channel_convs[i](combined_features)
            out_features.append(adjusted_features)
        return out_features

class samhierablock(nn.Module):
    def __init__(self, c1=144, c2=1, n=1):
        super(samhierablock, self).__init__()
        model_cfg = r"sam2_hiera_l.yaml"
        # 构建模型
        self.model = build_sam2(model_cfg, None, device="cpu").image_encoder.trunk
        # 替换 PatchEmbed 为 Identity
        self.model.patch_embed = nn.Identity()
        # 使模型参数不可训练
        #self.model = self.Model.trunk
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 获取模型的输出特征
        x=x.permute(0,2,3,1)
        features = self.model(x)[-1]
        return features

class samhierablockc2f(nn.Module):
    def __init__(self, c1=144, c2=1, n=1):
        super(samhierablockc2f, self).__init__()

        model_cfg = r"orgline/sam2_configs/sam2_hiera_l.yaml"

        # 构建模型
        self.model = build_sam2(model_cfg, None, device="cpu").image_encoder.trunk
        # 替换 PatchEmbed 为 Identity
        self.model.patch_embed = nn.Identity()
        # 使模型参数不可训练
        #self.model = self.Model.trunk
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 获取模型的输出特征
        x=x.permute(0,2,3,1)
        features = self.model(x)[-1]
        return features

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def hiera(weights='', **kwargs):
    model =samhiera()
    #print(model)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['model']))
    return model

def hiera2(weights='', **kwargs):
    model =samhiera2()
    #print(model)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['model']))
    return model

def hiera3(weights='', **kwargs):
    model =samhiera3()
    #print(model)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['model']))
    return model

def hiera3_tiny(weights='', **kwargs):
    model =samhiera3_tiny()
    #print(model)
    if weights:
        model.load_state_dict(update_weight(model.state_dict(), torch.load(weights)['model']))
    return model

if __name__ == "__main__":
    img = torch.rand(2, 3, 640, 640)
    import torch
    import torch.nn as nn
    model=hiera3_tiny()