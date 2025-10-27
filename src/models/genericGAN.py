# genericGAN.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_gan(epochs=50, batch_size=64, latent_dim=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # Use project data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = Generator(latent_dim)
    discriminator = Discriminator()

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.view(real_imgs.size(0), -1)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z).detach()
            real_loss = criterion(discriminator(real_imgs), real_labels)
            fake_loss = criterion(discriminator(fake_imgs), fake_labels)
            d_loss = real_loss + fake_loss

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(fake_imgs), real_labels)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save to project models directory
    model_path = os.path.join(os.path.dirname(__file__), "MnistGAN.pth")
    torch.save(generator.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# GUI interface functions
def get_constraints():
    return [f"Digit {i} %" for i in range(10)]

def generate_document(self, constraints, output_folder):
    import torch
    import numpy as np
    from PIL import Image
    import os

    ratios = [float(constraints.get(f"Digit {i} %", 0)) for i in range(10)]
    total = sum(ratios)
    if total == 0:
        raise ValueError("At least one digit ratio must be greater than 0.")
    probabilities = [r / total for r in ratios]
    digit_labels = np.random.choice(range(10), size=100, p=probabilities)

    images = []
    for i, label in enumerate(digit_labels):
        z = torch.randn(1, self.latent_dim)
        img = self.generator(z).detach().numpy().reshape(28, 28)
        img = ((img + 1) * 127.5).astype(np.uint8)
        images.append((label, Image.fromarray(img)))

    # Save composite image
    grid = Image.new("L", (280, 280))
    for idx, (_, img) in enumerate(images[:100]):
        x = (idx % 10) * 28
        y = (idx // 10) * 28
        grid.paste(img, (x, y))

    image_path = os.path.join(output_folder, "generated_digits.jpg")
    grid.save(image_path)

    # Create summary
    counts = {str(i): digit_labels.tolist().count(i) for i in range(10)}
    summary = "Synthetic Digit Generation Summary:\n"
    for digit, count in counts.items():
        summary += f"Digit {digit}: {count} samples\n"
    summary += f"\nImage saved as: {image_path}"
    return summary


# Train when run directly
if __name__ == "__main__":
    train_gan()
