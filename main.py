import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tensorflow as tf
import math
import time
from FSRCNN import FSRCNN, ImageDataset

url = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/'
train_lr_path = "DIV2K_train_LR_unknown_X2.zip"
valid_lr_path = "DIV2K_valid_LR_unknown_X2.zip"
train_hr_path = "DIV2K_train_HR.zip"
valid_hr_path = "DIV2K_valid_HR.zip"

tf.keras.utils.get_file(train_lr_path, url + train_lr_path, extract=True, cache_dir=".")
tf.keras.utils.get_file(train_hr_path, url + train_hr_path, extract=True, cache_dir=".")
tf.keras.utils.get_file(valid_lr_path, url + valid_lr_path, extract=True, cache_dir=".")
tf.keras.utils.get_file(valid_hr_path, url + valid_hr_path, extract=True, cache_dir=".")

train_lr_path = "datasets/DIV2K_train_LR_unknown/X2"
train_hr_path = "datasets/DIV2K_train_HR"
valid_lr_path = "datasets/DIV2K_valid_LR_unknown/X2"
valid_hr_path = "datasets/DIV2K_valid_HR"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define parameters
learning_rate = 0.0001
num_epochs = 100
batch_size = 1


# Initalize model and optimizer
model = FSRCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load training and validation datasets
train_dataset = ImageDataset(train_lr_path, train_hr_path)
val_dataset = ImageDataset(valid_lr_path, valid_hr_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

# Define loss function
loss_fn = nn.MSELoss()

#Check if model exists already and load it if so
if os.path.exists("fsrcnn.pth"):
    model.load_state_dict(torch.load("fsrcnn.pth", map_location=device))

# Train and validate model
for epoch in range(num_epochs):
    start_time = time.time()

    # Train model
    model.train()
    train_loss = 0
    train_psnr = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = loss_fn(outputs.float(), targets.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # Calculate PSNR
        mse = loss.item()
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        train_psnr += psnr

    train_loss /= len(train_loader)
    train_psnr /= len(train_loader)

    # Validate model
    model.eval()
    val_loss = 0
    val_psnr = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            loss = loss_fn(outputs.float(), targets.float())
            val_loss += loss.item()
            # Calculate PSNR
            mse = loss.item()
            psnr = 20 * math.log10(1.0 / math.sqrt(mse))
            val_psnr += psnr

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)

    end_time = time.time()

    # Log epoch metrics
    epoch_time = end_time - start_time
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.4f}, Epoch Time: {epoch_time:.2f} seconds')

    # Save model weights
    torch.save(model.state_dict(), 'fsrcnn.pth')