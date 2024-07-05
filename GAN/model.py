import torch
import torch.nn as nn
from torch import Tensor

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "Discriminator_64_G", "Generator_64_G",
    "Discriminator_64_RGB", "Generator_64_RGB",
    "Discriminator_256_G", "Generator_256_G",
    "Discriminator_256_RGB", "Generator_256_RGB",
    "Discriminator_512_G", "Generator_512_G",
    "Discriminator_512_RGB", "Generator_512_RGB"
]



class Discriminator_64_G(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_64_G, self).__init__()
        self.main = nn.Sequential(
            # Input is 1 x 64 x 64
            nn.Conv2d(1, 64, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 32 x 32
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 16 x 16
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 8 x 8
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 4 x 4
            nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        out = torch.flatten(out, 1)

        return out

class Generator_64_G(nn.Module):
    def __init__(self) -> None:
        super(Generator_64_G, self).__init__()
        self.main = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 1, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
            # state size. 1 x 64 x 64
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)


class Discriminator_64_RGB(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_64_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 3 x 64 x 64
            nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 32 x 32
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 16 x 16
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 8 x 8
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 4 x 4
            nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        # print(f"Discriminator_64_RGB: {out.size()}")
        out = torch.flatten(out, 1)

        return out

class Generator_64_RGB(nn.Module):
    def __init__(self) -> None:
        super(Generator_64_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)


class Discriminator_256_RGB(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_256_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 1 x 256 x 256
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 16 x 16
            #################################################  New Layers  #################################################
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            ################################################################################################################
            # State size. 512 x 4 x 4
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=True)
        # self.conv2 = nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv5 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv6 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn6 = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True)
        # self.lr = nn.LeakyReLU(0.2, True)
        # self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        # print(f"Discriminator_512_RGB: {out.size()}")
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.conv1(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn2(self.conv2(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn3(self.conv3(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn4(self.conv4(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn5(self.conv5(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn6(self.conv6(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.sigmoid(self.conv7(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        out = torch.flatten(out, 1)

        return out

class Generator_256_G(nn.Module):
    def __init__(self) -> None:
        super(Generator_256_G, self).__init__()
        self.main = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            #################################################  New Layers  #################################################
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 16 x 16
            ################################################################################################################
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 32 x 32
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 64 x 64
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 128 x 128
            nn.ConvTranspose2d(64, 1, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
            # state size. 3 x 256 x 256
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)


class Discriminator_256_RGB(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_256_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 3 x 256 x 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 128 x 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 64 x 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 16 x 16
            #################################################  New Layers  #################################################
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            ################################################################################################################
            # State size. 512 x 4 x 4
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=True)
        # self.conv2 = nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv5 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv6 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn6 = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True)
        # self.lr = nn.LeakyReLU(0.2, True)
        # self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        # print(f"Discriminator_512_RGB: {out.size()}")
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.conv1(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn2(self.conv2(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn3(self.conv3(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn4(self.conv4(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn5(self.conv5(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn6(self.conv6(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.sigmoid(self.conv7(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        out = torch.flatten(out, 1)

        return out

class Generator_256_RGB(nn.Module):
    def __init__(self) -> None:
        super(Generator_256_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            #################################################  New Layers  #################################################
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 16 x 16
            ################################################################################################################
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 32 x 32
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 64 x 64
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 128 x 128
            nn.ConvTranspose2d(64, 3, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
            # state size. 3 x 256 x 256
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)


class Discriminator_512_G(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_512_G, self).__init__()
        self.main = nn.Sequential(
            # Input is 3 x 512 x 512
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 256 x 256
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 128 x 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 64 x 64
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 32 x 32
            #################################################  New Layers  #################################################
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            ################################################################################################################
            # State size. 512 x 1 x 1 -> 512 x 4 x 4
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=True)
        # self.conv2 = nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv5 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv6 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn6 = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True)
        # self.lr = nn.LeakyReLU(0.2, True)
        # self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        # print(f"Discriminator_512_RGB: {out.size()}")
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.conv1(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn2(self.conv2(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn3(self.conv3(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn4(self.conv4(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn5(self.conv5(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn6(self.conv6(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.sigmoid(self.conv7(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        out = torch.flatten(out, 1)

        return out

class Generator_512_G(nn.Module):
    def __init__(self) -> None:
        super(Generator_512_G, self).__init__()
        self.main = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            #################################################  New Layers  #################################################
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 16 x 16
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 32 x 32
            ################################################################################################################
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 64 x 64
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 128 x 128
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 256 x 256
            nn.ConvTranspose2d(64, 1, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
            # state size. 1 x 512 x 512
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)


class Discriminator_512_RGB(nn.Module):
    def __init__(self) -> None:
        super(Discriminator_512_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 3 x 512 x 512
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 256 x 256
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 128 x 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 64 x 64
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 32 x 32
            #################################################  New Layers  #################################################
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 16 x 16
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 8 x 8
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            ################################################################################################################
            # State size. 512 x 1 x 1 -> 512 x 4 x 4
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=True),
            nn.Sigmoid()
        )

        # self.conv1 = nn.Conv2d(3, 64, (4, 4), (2, 2), (1, 1), bias=True)
        # self.conv2 = nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv5 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.conv6 = nn.Conv2d(512, 512, (4, 4), (2, 2), (1, 1), bias=False)
        # self.bn6 = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True)
        # self.lr = nn.LeakyReLU(0.2, True)
        # self.sigmoid = nn.Sigmoid()
        

    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        # print(f"Discriminator_512_RGB: {out.size()}")
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.conv1(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn2(self.conv2(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn3(self.conv3(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn4(self.conv4(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn5(self.conv5(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.lr(self.bn6(self.conv6(x)))
        # print(f"Discriminator_512_RGB: {x.size()}")
        # x = self.sigmoid(self.conv7(x))
        # print(f"Discriminator_512_RGB: {x.size()}")
        out = torch.flatten(out, 1)

        return out

class Generator_512_RGB(nn.Module):
    def __init__(self) -> None:
        super(Generator_512_RGB, self).__init__()
        self.main = nn.Sequential(
            # Input is 100, going into a convolution.
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            #################################################  New Layers  #################################################
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 8 x 8
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 16 x 16
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 32 x 32
            ################################################################################################################
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 64 x 64
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 128 x 128
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 256 x 256
            nn.ConvTranspose2d(64, 3, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Tanh()
            # state size. 3 x 512 x 512
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support PyTorch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.main(x)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)
