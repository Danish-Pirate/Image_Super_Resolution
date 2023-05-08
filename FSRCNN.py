import torch.nn as nn
import os
from PIL import Image
from glob import glob
from torchvision.transforms import transforms
import torch

## Defining the FSRCNN model
class FSRCNN(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=3):
        super(FSRCNN, self).__init__()

        # Feature extraction layer.
        self.extract = nn.Sequential(
        nn.Conv2d(in_channels=num_channels, out_channels=56, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        nn.PReLU(num_parameters=56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=12, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.PReLU(num_parameters=12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.PReLU(num_parameters=12),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.PReLU(num_parameters=12),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.PReLU(num_parameters=12),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.PReLU(num_parameters=12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.PReLU(num_parameters=56)
        )

        # Deconvolution layer.
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=56, out_channels=num_channels, kernel_size=(9, 9), stride=(upscale_factor, upscale_factor), padding=(4, 4), output_padding=(upscale_factor - 1, upscale_factor - 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.extract(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x


class ImageDataset:
    def __init__(self, lr_directory: str, hr_directory: str):
        self.lr_files: list[str] = sorted(glob(os.path.join(lr_directory, "*.png")))
        self.hr_files: list[str] = sorted(glob(os.path.join(hr_directory, "*.png")))
        self.length = len(self.hr_files)

    def __len__(self):
        return self.length
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    def __getitem__(self, item_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(self.lr_files[item_id], "rb") as lr_fh:
            with Image.open(lr_fh) as lr_image:
                lr_data = self.transform(lr_image)

        with open(self.hr_files[item_id], "rb") as hr_fh:
            with Image.open(hr_fh) as hr_image:
                hr_data = self.transform(hr_image)

        return lr_data, hr_data