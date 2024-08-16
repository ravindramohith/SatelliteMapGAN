
# BiCycleGAN - Satellite to Map Image Translation

## Overview
This project implements the **BiCycleGAN** architecture for multimodal image-to-image translation from scratch using PyTorch. The BiCycleGAN model is particularly suited for tasks where multiple plausible outputs exist for a given input image. This project specifically focuses on translating satellite images to their corresponding Google Maps images and vice versa.

### What is BiCycleGAN?
BiCycleGAN is a hybrid model that combines the strengths of **cVAE-GAN** (Conditional Variational Autoencoder GAN) and **cLR-GAN** (Conditional Latent Regressor GAN). The cVAE-GAN helps in generating diverse outputs by sampling from a Gaussian latent space, while the cLR-GAN ensures the consistency and accuracy of these outputs by regressing the latent code back from the generated image.

## Dataset
### Satellite to Map Translation Dataset
The dataset consists of satellite images of New York and their corresponding Google Maps images. It is structured as follows:
- **Training Set**: 1,097 images
- **Validation Set**: 1,099 images

Each image in the dataset is 1,200 pixels wide and 600 pixels tall, with the satellite view on the left and the map view on the right.

## Implementation Details

### 1. Data Preprocessing and Augmentation
The data preprocessing involves:
- **Image Splitting**: Each image is split into two parts: the left half as the satellite image and the right half as the map image.
- **Data Augmentation**: Random horizontal flipping is applied to both satellite and map images with a 50% probability. This augmentation helps in increasing the diversity of the training data, making the model more robust.

### 2. Model Architecture
The core of the BiCycleGAN implementation involves several key components:

#### a. U-Net Generator
- **U-Net Architecture**: The generator is based on a U-Net architecture, which allows it to capture both high-level and low-level features, making it effective for tasks where the input and output images have a similar structure.
- **Skip Connections**: These connections between the encoder and decoder layers help in retaining spatial information, leading to sharper and more accurate image generation.

#### b. PatchGAN Discriminators
- **Discriminator Architecture**: The model employs PatchGAN discriminators, which classify whether 70x70 image patches are real or fake. This ensures that both global structures and finer details in the image are well captured.
- **Multi-Scale Discriminators**: Multiple discriminators operating at different scales are used to ensure both local and global consistency of the generated images.

### 3. Latent Space Engineering
- **Gaussian Latent Space**: A Gaussian latent space is engineered for the model, enabling it to produce a diverse set of outputs for a single input image. This is done by sampling latent vectors from a Gaussian distribution during training.
- **Stochastic Sampling**: During the generation process, stochastic sampling of latent codes allows the model to explore different modes of the output distribution, leading to varied yet realistic image translations.

### 4. Loss Functions
The training process of the BiCycleGAN is guided by several loss functions:
- **Adversarial Loss (GAN Loss)**: This loss drives the generator to produce images that are indistinguishable from real images, as judged by the discriminator.
- **L1 Loss**: This loss is used to minimize the pixel-wise difference between the generated image and the ground truth image, ensuring that the generated image closely matches the target.
- **KL Divergence Loss (KL Loss)**: The KL divergence loss is applied to enforce that the latent code distribution approximates a standard Gaussian distribution, which is crucial for meaningful sampling from the latent space.

### 5. Training Procedure
- **Data Handling**: Efficient data loading and augmentation techniques are employed to handle the large dataset without bottlenecks.
- **Training**: The model is trained with a combination of the aforementioned loss functions, ensuring that it learns both the global structure and fine details of the images. The training is monitored with real-time loss plots and image outputs.

## Code Walkthrough

### 1. Imports and Setup
The notebook starts with importing necessary libraries such as PyTorch, torchvision for data handling, PIL for image processing, and matplotlib for visualization.

```python
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
```

### 2. DataLoader Setup
A custom dataset class is implemented to handle the loading and preprocessing of the images. The class splits each image into its satellite and map components, applies the necessary transformations, and supports on-the-fly data augmentation.

```python
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        img_A = img.crop((0, 0, img.width // 2, img.height))
        img_B = img.crop((img.width // 2, 0, img.width, img.height))
        
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {"A": img_A, "B": img_B}
    
    def __len__(self):
        return len(self.files)
```

### 3. Generator and Discriminator Definitions
The generator is implemented using a U-Net architecture with skip connections, while the PatchGAN discriminators are designed to operate on 70x70 patches of the image. This allows the model to focus on both fine details and overall image structure.

```python
class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetGenerator, self).__init__()
        # Define encoder and decoder layers
        
    def forward(self, x):
        # Implement the forward pass with skip connections
        return output

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        # Define layers
        
    def forward(self, x):
        # Implement the forward pass
        return output
```

### 4. Loss Function Implementations
The loss functions are carefully crafted to guide the training of the BiCycleGAN, ensuring that the model generates high-quality images that are both diverse and accurate.

```python
adversarial_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
kl_loss = nn.KLDivLoss()
```

### 5. Training Loop
The training loop integrates all components, including data loading, model forward passes, loss computation, and backpropagation. Real-time monitoring of the losses and visual outputs is performed to ensure the model is learning effectively.

```python
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        # Load batch data
        # Forward pass through generator and discriminator
        # Compute losses
        # Backpropagation and optimizer step
        
        # Print losses and save images for monitoring
```

### 6. Results and Evaluation
After training, the model is evaluated on the validation dataset. The results include both qualitative visualizations of generated images and quantitative analysis of the losses.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)
- matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ravindramohith/SatelliteMapGAN.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset as described above.

4. Run the Jupyter notebook to start training:
   ```bash
   jupyter notebook bicyclegan.ipynb
   ```

## Results
The model successfully learns to translate satellite images into maps and vice versa. The results demonstrate the model's ability to generate multiple plausible outputs for a single input image, showcasing the power of multimodal image-to-image translation.

### Sample Results
- Generated maps from satellite images with high fidelity.
- Diverse outputs generated from stochastic sampling of the latent space.

## Conclusion
This project demonstrates the effective application of BiCycleGAN for satellite-to-map image translation, achieving high-quality results with diverse outputs. The combination of cVAE-GAN and cLR-GAN, along with U-Net generators and PatchGAN discriminators, proves to be powerful for multimodal image generation tasks.

## Acknowledgments
This implementation is inspired by the BiCycleGAN paper titled **"Toward Multimodal Image-to-Image Translation"**. The dataset used is provided by the pix2pix repository.