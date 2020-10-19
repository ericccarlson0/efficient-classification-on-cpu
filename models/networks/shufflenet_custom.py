import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups):
    batches, channels, h, w = x.data.size()
    c_per_g = channels // groups

    x = x.view(batches, groups, c_per_g, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batches, -1, h, w)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (0 < stride < 4):
            raise ValueError(f"Illegal stride of {stride}.")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: torch.Tensor):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out

# NOTE: stages_repeats are [4, 8, 4] and stages_out_channels are [24, 48, 96, 192, 1024] for x0.5
# and are [24, 116, 232, 464, 1024] for x1.0
class ShuffleNetCustom(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=2):
        super(ShuffleNetCustom, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("Expected stages_repeats to contain three positive integers.")
        if len(stages_out_channels) != 5:
            raise ValueError("Expected stages_out_channels to contain five positive integers.")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]

        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            sequence = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                sequence.append(InvertedResidual(output_channels, output_channels, 1))

            setattr(self, name, nn.Sequential(*sequence))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # GLOBAL POOL
        x = self.fc(x)
        return x

def shufflenet_small(num_classes: int = 2):
    return ShuffleNetCustom([2, 4, 4], [24, 48, 96, 192, 1024], num_classes=num_classes)

def prepare_for_finetune(net: ShuffleNetCustom, depth: int = 0):
    for param in net.parameters():
        param.requires_grad = False

    modules = [net.fc, net.conv5, net.stage4, net.stage3, net.stage2, net.conv1]
    for i, module in enumerate(modules, 1):
        if depth < i:
            break
        for param in module:
            param.requires_grad = True

    return net
