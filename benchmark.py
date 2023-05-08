from FSRCNN import FSRCNN, ImageDataset
import torch
import torch.nn as nn
import math
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FSRCNN().to(device)

test_lr_path = "Set5/lr"
test_hr_path = "Set5/hr"

FSRCNN_output = "benchmark_output/FSRCNN"
bicubic_output = "benchmark_output/bicubic"
bilinear_output = "benchmark_output/bilinear"
nearest_neighbour_output = "benchmark_output/nearest_neighbour"

# Load test dataset
test_dataset = ImageDataset(test_lr_path, test_hr_path)

# Load training and validation datasets
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# load trained model
model.load_state_dict(torch.load("fsrcnn.pth", map_location="cpu"))

# set the model to evaluation mode
model.eval()

# Loss function late used to calculate PSNR
loss_fn = nn.MSELoss()

# Define functions
def calc_PSNR(output_data, target_data):
    loss = loss_fn(output_data, target_data)
    mse = loss.item()
    PSNR = 20 * math.log10(1.0 / math.sqrt(mse))
    return PSNR

def fsrcnn(input_data, target_data):
    # Pass the input through the FSRCNN model
    output_data = model(input_data.float())

    # Calculate the PSNR value and print it
    PSNR = calc_PSNR(output_data, target_data)
    print(f"Image {image_number} PSNR using FSCRNN: {PSNR}")
    
    # Save the image
    save_image(output_data, f'{FSRCNN_output}/image{image_number}.png')

def bicubic(input_data, target_data):
     # Resize the input tensor using bicubic interpolation
    output_data = torch.nn.functional.interpolate(input_data, size=target_data.shape[-2:], mode='bicubic', align_corners=True)

    # Calculate the PSNR value and print it
    PSNR = calc_PSNR(output_data, target_data)
    print(f"Image {image_number} PSNR using Bicubic: {PSNR}")
    
    # Save the image
    save_image(output_data, f'{bicubic_output}/image{image_number}.png')

def bilinear(input_data, target_data):
     # Resize the input tensor using bicubic interpolation
    output_data = torch.nn.functional.interpolate(input_data, size=target_data.shape[-2:], mode='bilinear', align_corners=True)

    # Calculate the PSNR value and print it
    PSNR = calc_PSNR(output_data, target_data)
    print(f"Image {image_number} PSNR using Bilinear: {PSNR}")
    
    # Save the image
    save_image(output_data, f'{bilinear_output}/image{image_number}.png')

def nearest_neighbour(input_data, target_data):
     # Resize the input tensor using bicubic interpolation
    output_data = torch.nn.functional.interpolate(input_data, size=target_data.shape[-2:], mode='nearest')

    # Calculate the PSNR value and print it
    PSNR = calc_PSNR(output_data, target_data)
    print(f"Image {image_number} PSNR using Nearest-nearbour: {PSNR}")
    
    # Save the image
    save_image(output_data, f'{nearest_neighbour_output}/image{image_number}.png')

# disable gradients during testing
image_number = 1
with torch.no_grad():
    for i, (input_data, target_data) in enumerate(test_loader):
        input_data, target_data = input_data.to(device), target_data.to(device)
        
        # using FSRCNN
        fsrcnn(input_data, target_data)

        # using Bicubic
        bicubic(input_data, target_data)

        # using Bilnear
        bilinear(input_data, target_data)

        # using Nearest-neighbour
        nearest_neighbour(input_data, target_data)
        image_number = image_number + 1
