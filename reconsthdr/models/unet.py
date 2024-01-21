import torch
from torch import nn


class Unet(nn.Module):
    def __init__(self, in_channel: int=3) -> None:
        super().__init__()
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down_layers1 = UnetDonwLayer(64, 128)
        self.down_layers2 = UnetDonwLayer(128, 256)
        self.down_layers3 = UnetDonwLayer(256, 512)
        self.down_layers4 = UnetDonwLayer(512, 1024)
        self.up_layers1 = UnetUpLayer(1024, 512)
        self.up_layers2 = UnetUpLayer(512, 256)
        self.up_layers3 = UnetUpLayer(256, 128)
        self.up_layers4 = UnetUpLayer(128, 128)
        self.last_layer = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x: torch.Tensor):
        x1 = self.init_layer(x)  # (bs, 64, h, w)
        x2 = self.down_layers1(x1)  # (bs, 128, h/2, w/2)
        x3 = self.down_layers2(x2)  # (bs, 256, h/4, w/4)
        x4 = self.down_layers3(x3)  # (bs, 512, h/8, w/8)
        x5 = self.down_layers4(x4)  # (bs, 1024, h/16, w/16)
        x = self.up_layers1(x5, x4)  # (bs, 512, h/8, w/8)
        x = self.up_layers2(x, x3)  # (bs, 256, h/4, w/4)
        x = self.up_layers3(x, x2)  # (bs, 128, h/2, w/2)
        x = self.up_layers4(x, x1)  # (bs, 64, h, w)
        return self.last_layer(x)


class UnetDonwLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers: nn.Sequential = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor):
        return self.layers(x)

class UnetUpLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsampling_layer = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor):
        x = self.upsampling_layer(x)
        x = torch.cat([x, skip_x], dim=1)
        return self.layers(x)


if __name__ == "__main__":
    img = torch.rand(1, 3, 256, 256)
    model = Unet()
    feature = model(img)
    print(feature.shape)
