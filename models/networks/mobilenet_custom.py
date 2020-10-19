import numpy as np

import torch
import torch.nn as nn

def conv_bn(inp: int, oup: int, stride: int, nonlinear: nn.Module = nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nonlinear(inplace=True)
    )

def pw_conv_bn(inp: int, oup: int, nonlinear: nn.Module = nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nonlinear(inplace=True)
    )

def make_divisible(x, divisor: int=8):
    return int(np.ceil(x * 1. / divisor) * divisor)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        return x * nn.functional.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        return nn.functional.relu6(x + 3., inplace=self.inplace) / 6.


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor):
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileBottleneck(nn.Module):

    def __init__(self, inp: int, oup: int, kernel: int, stride: int,
                 exp: int, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()

        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            nonlinear = nn.ReLU
        elif nl == 'HS':
            nonlinear = Hswish
        else:
            raise NotImplementedError

        if se:
            se_layer = SEModule
        else:
            se_layer = Identity

        self.conv = nn.Sequential(
            # pointwise conv.
            nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp),
            nonlinear(inplace=True),
            # depthwise conv.
            nn.Conv2d(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            se_layer(exp),
            nonlinear(inplace=True),
            # pointwise-linear
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_classes: int = 1000, input_size: int = 224, dropout: float = 0.8,
                 mode='small', w_multiplier: float = 1.0):
        super(MobileNetV3, self).__init__()
        assert input_size % 32 == 0

        input_channel = 16
        last_channel = 1280

        # Produces layer representations as lists of [k, exp, c, se, nl, s]
        if mode == 'large':
            setting = LARGE_SETTING
        elif mode == 'small':
            setting = SMALL_SETTING
        else:
            raise NotImplementedError

        # FIRST layer.
        if w_multiplier > 1.0:
            last_channel = make_divisible(last_channel * w_multiplier)
        self.features = [conv_bn(3, input_channel, 2, nonlinear=Hswish)]
        self.classifier = []

        # MOBILE blocks.
        for k, exp, c, se, nl, s in setting:
            # TODO: what is up with these make_divisible?
            output_channel = make_divisible(c * w_multiplier)
            exp_channel = make_divisible(exp * w_multiplier)

            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # LAST layers.
        if mode == 'large':
            last_conv = make_divisible(960 * w_multiplier)
        elif mode == 'small':
            last_conv = make_divisible(576 * w_multiplier)
        else:
            raise NotImplementedError

        self.features.append(pw_conv_bn(input_channel, last_conv, nonlinear=Hswish))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        self.features.append(Hswish(inplace=True))

        # Finish set up.
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, n_classes)
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # Use custom initialization...
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, .01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def mobilenet_small(pretrained=False, net_dir='', **kwargs):
    net = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load(net_dir,
                                map_location=torch.device('cpu'))
        net.load_state_dict(state_dict, strict=True)

    return net

def activate_grad(module: nn.Module):
    # Test to make sure that the correct module/layer is activated.
    # print(module)

    for param in module.parameters():
        param.requires_grad = True

# TODO: This only works for the SMALL version at the moment.
def prepare_for_finetune(net: MobileNetV3, depth: int = 0):
    # This loads the entire generator... could be more efficient but that would require reversing the generator.
    layers = list(net.features.children())
    for param in net.parameters():
        param.requires_grad = False

    if depth >= 1:
        activate_grad(net.classifier)
    if depth >= 2:
        for i in [14, 15]:
            activate_grad(layers[i])
    if depth >= 3:
        for i in [12, 13]:
            activate_grad(layers[i])

    # "offset" accounts for the difference between the specified depth and the depth of the bottleneck layers in
    # MobileNetV3.features
    offset = 1
    for i in range(4, len(layers) - offset):
        if i > depth:
            break

        dex = i + offset
        activate_grad(layers[-dex])

    return net

# number bottlenecks = 15
LARGE_SETTING = [
    [3, 16, 16, False, 'RE', 1],
    [3, 64, 24, False, 'RE', 2],
    [3, 72, 24, False, 'RE', 1],
    [5, 72, 40, True, 'RE', 2],
    [5, 120, 40, True, 'RE', 1],
    [5, 120, 40, True, 'RE', 1],
    [3, 240, 80, False, 'HS', 2],
    [3, 200, 80, False, 'HS', 1],
    [3, 184, 80, False, 'HS', 1],
    [3, 184, 80, False, 'HS', 1],
    [3, 480, 112, True, 'HS', 1],
    [3, 672, 112, True, 'HS', 1],
    [5, 672, 160, True, 'HS', 2],
    [5, 960, 160, True, 'HS', 1],
    [5, 960, 160, True, 'HS', 1]
]

# number bottlenecks = 11
SMALL_SETTING = [
    [3, 16, 16, True, 'RE', 2],
    [3, 72, 24, False, 'RE', 2],
    [3, 88, 24, False, 'RE', 1],
    [5, 96, 40, True, 'HS', 2],
    [5, 240, 40, True, 'HS', 1],
    [5, 240, 40, True, 'HS', 1],
    [5, 120, 48, True, 'HS', 1],
    [5, 144, 48, True, 'HS', 1],
    [5, 288, 96, True, 'HS', 2],
    [5, 576, 96, True, 'HS', 1],
    [5, 576, 96, True, 'HS', 1]
]

if __name__ == '__main__':
    model = mobilenet_small(pretrained=True, net_dir='mobilenetv3_small.pth.tar')

    print(f'total parameters: {sum(p.numel() for p in model.parameters()) / 1_000_000: .3f}M')
