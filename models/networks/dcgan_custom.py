import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.print_tag = True

        self.stage0 = nn.Sequential(
            # filter -> 5
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )

        inp = ngf * 8
        channel_nums = [ngf * 4, ngf * 2, ngf]
        # stage1, stage2, stage3
        for i, channel_num in enumerate(channel_nums, 1):
            oup = channel_num
            stage = nn.Sequential(
                # filter -> 5
                # nn.ConvTranspose2d(inp, oup, 5, 2, 2, bias=False),
                nn.ConvTranspose2d(inp, oup, 4, 2, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
            inp = oup

            setattr(self, f"stage{i}", stage)

        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        for i in range(5):
            stage = getattr(self, f"stage{i}")
            x = stage(x)

            if self.print_tag:
                print(f"Generator stage{i}: ", x.shape)

        self.print_tag = False
        return x


class Discriminator(nn.Module):
    def __init__(self, nc: int, ndf: int):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.print_tag = True

        self.stage0 = nn.Sequential(
            # filter -> 5
            # nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        inp = ndf
        channel_nums = [ndf * 2, ndf * 4, ndf * 8]
        # stage1, stage2, stage3
        for i, channel_num in enumerate(channel_nums, 1):
            oup = channel_num
            stage = nn.Sequential(
                # filter -> 5
                # nn.Conv2d(inp, oup, 5, 2, 2, bias=False),
                nn.Conv2d(inp, oup, 4, 2, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(0.2, inplace=True)
            )
            inp = oup

            setattr(self, f"stage{i}", stage)

        self.stage4 = nn.Sequential(
            # filter -> 5
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        for i in range(5):
            stage = getattr(self, f"stage{i}")
            x = stage(x)

            if self.print_tag:
                print(f"Discriminator stage{i}: ", x.shape)

        self.print_tag = False
        return x
