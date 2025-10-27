import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import threading
import io

# ------------------ GAN Models ------------------
class Generator(nn.Module):
    def __init__(self, z_dim=128, img_dim=784):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# ------------------ GUI Setup ------------------
root = tk.Tk()
root.title("GAN Training Viewer")

canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ------------------ Training Setup ------------------
generator = Generator()
discriminator = Discriminator()
lossBCE = nn.BCELoss()
gOptimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
dOptimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))
])

mnistLoader = DataLoader(
    datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=100,
    shuffle=True
)

gLosses = []
dLosses = []

# ------------------ Helper: Convert Matplotlib Figure to Tk Image ------------------
def fig_to_image(fig):
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf)
    return ImageTk.PhotoImage(img)

# ------------------ Training Thread ------------------
def train():
    for epoch in range(50):
        for realImages, _ in mnistLoader:
            noise = torch.randn(realImages.size(0), 128)
            fakeImages = generator(noise)

            realPreds = discriminator(realImages)
            fakePreds = discriminator(fakeImages.detach())

            realLoss = lossBCE(realPreds, torch.ones_like(realPreds))
            fakeLoss = lossBCE(fakePreds, torch.zeros_like(fakePreds))
            dLoss = realLoss + fakeLoss

            dOptimizer.zero_grad()
            dLoss.backward()
            dOptimizer.step()

            preds = discriminator(fakeImages)
            gLoss = lossBCE(preds, torch.ones_like(preds))

            gOptimizer.zero_grad()
            gLoss.backward()
            gOptimizer.step()

        gLosses.append(gLoss.item())
        dLosses.append(dLoss.item())

        print(f"Epoch [{epoch+1}/50] | dLoss: {dLoss.item():.4f} | gLoss: {gLoss.item():.4f}")

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                noise = torch.randn(16, 128)
                fakeImages = generator(noise).view(-1, 1, 28, 28)
                fakeImages = (fakeImages + 1) / 2

                fig, axes = plt.subplots(4, 4, figsize=(6, 6))
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(fakeImages[i].squeeze(), cmap="gray")
                    ax.axis("off")
                plt.tight_layout()

                img = fig_to_image(fig)
                label = tk.Label(scrollable_frame, image=img)
                label.image = img
                label.pack(pady=10)

                text = tk.Label(scrollable_frame, text=f"Epoch {epoch+1} | dLoss: {dLoss.item():.4f} | gLoss: {gLoss.item():.4f}")
                text.pack()

                plt.close(fig)

    # Final loss plot
    fig = plt.figure(figsize=(10, 4))
    plt.plot(gLosses, label="Generator Loss")
    plt.plot(dLosses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)

    img = fig_to_image(fig)
    label = tk.Label(scrollable_frame, image=img)
    label.image = img
    label.pack(pady=10)
    plt.close(fig)

# ------------------ Start Training ------------------
threading.Thread(target=train, daemon=True).start()
root.mainloop()
