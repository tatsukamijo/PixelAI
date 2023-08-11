import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import re
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F
from scipy import ndimage


class Conv_decoder(nn.Module):
    def __init__(self):
        super(Conv_decoder, self).__init__()

        # Two fully connected layers of neurons (feedforward architecture)
        self.ff_layers=nn.Sequential( 
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 7 * 10 * 16),  # 1120 neurons
            nn.ReLU(),
        )

        # Sequential upsampling using the deconvolutional layers & smoothing out checkerboard artifacts with conv layers
        self.conv_layers=nn.Sequential(
            nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1), #deconv1
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), #conv1
            nn.ReLU(),
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1), #deconv2
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1), #conv2
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), #deconv3
            nn.Sigmoid() #Squeezing the output to 0-1 range
        )


    def forward(self, x):
        x = self.ff_layers(x)        
        x = x.view(-1, 16, 7, 10) # Reshaping the output of the fully connected layers so that it is compatible with the conv layers
        x = self.conv_layers(x)
        return x


# Custom Dataset class
class TactileRGBDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.file_list = os.listdir(directory)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_path = os.path.join(self.directory, file_name)

        # Extracting the float number from the filename
        float_number = float(re.findall(r'(\d+\.\d+)(?=\.)', file_name)[0])
        
        base_image_path = 'notouch_tactile_left.png'

        # Loading the image
        image = Image.open(image_path)
        image = np.array(image)  
        img = cv2.subtract(image, cv2.imread(base_image_path))
        
        img = transforms.ToTensor()(Image.fromarray(img))

        img = img - torch.min(img)
        img = img / torch.max(img)
        img = F.adjust_brightness(
                    F.adjust_contrast(
                        img,
                        contrast_factor=4.5,
                    ),
                    brightness_factor=7,
                )
        # print(img.shape)
        # to img
        img = transforms.ToPILImage(mode="RGB")(img)

        # Get the pixels
        pixels = img.load()

        # Iterate through the image pixels
        for i in range(img.width):
            for j in range(img.height):
                r, g, b = pixels[i, j]
                if r + g + b < 200:
                    pixels[i, j] = (0, 0, 0)  # Make the pixel black
                else:
                    pixels[i, j] = (255, 255, 255)  # Make the pixel white

        # convert to grayscale
        img = img.convert('L')
        img_np = np.array(img)  # Convert PIL Image to NumPy array
        img_np = ndimage.zoom(img_np, (0.5, 0.467), order=3)  # 0.5 means 50% of original size
        # transpose
        img_np = img_np.transpose(1, 0)
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        # print(img_tensor.shape)

        return float_number, img_tensor
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

    
# Parameters
directory = '/PixelAI/tactile'
batch_size = 32
epochs = 1000
learning_rate = 0.0001

wandb.init(project='tactile_decoder')
wandb.config.batch_size = batch_size
wandb.config.epochs = epochs
wandb.config.learning_rate = learning_rate

# Loading the dataset
dataset = TactileRGBDataset(directory)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
model = Conv_decoder().to(device)
model.train()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    for batch_idx, (float_numbers, images) in enumerate(dataloader):
        float_numbers = float_numbers.view(-1, 1).float().to(device)
        images = images.to(device)

        # Forward pass
        outputs = model(float_numbers)
        loss = criterion(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}')
        wandb.log({'loss': loss.item()})
    # Save the model per epoch
    torch.save(model.state_dict(), f'model/decoder_{epoch}.pth')

    # test the model
    model.eval()
    with torch.no_grad():
        for batch_idx, (float_numbers, images) in enumerate(dataloader):
            float_numbers = float_numbers.view(-1, 1).float().to(device)
            images = images.to(device)

            # Forward pass
            outputs = model(float_numbers)
            loss = criterion(outputs, images)

            # visualize the output
            # plt.imshow(outputs[0].squeeze().cpu().detach().numpy().transpose(1, 0), cmap='gray')
            # plt.title('Output')
            # plt.show()

            # log output image with wandb
            wandb.log({'output': [wandb.Image(outputs[0].squeeze().cpu().detach().numpy().transpose(1, 0), caption='Output')]})

            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}')

            break

# Saving the model
torch.save(model.state_dict(), 'decoder_modified.pth')

print('Training complete.')



