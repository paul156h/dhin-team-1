import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import pandas as pd
import random

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "ccdGAN.pth"
MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME)

LATENT_DIM = 100
OUTPUT_DIM = 4  # CCD-style vector: [age, gender, has_diabetes, num_meds]
EPOCHS = 500
BATCH_SIZE = 64

# === GAN Architecture ===
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=OUTPUT_DIM):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Synthetic CCD Generator (Sandboxed per epoch) ===
def generate_sandboxed_ccd_vectors(batch_size=BATCH_SIZE):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ccd_batch.csv")
        data = []

        for _ in range(batch_size):
            age = random.randint(0, 100) / 100
            gender = random.choice([0, 1])
            has_diabetes = random.choice([0, 1])
            num_meds = random.randint(0, 10) / 10
            data.append([age, gender, has_diabetes, num_meds])

        df = pd.DataFrame(data, columns=["age", "gender", "diabetes", "meds"])
        df.to_csv(path, index=False)

        loaded = pd.read_csv(path).values
        sample_count = loaded.shape[0]
        print(f"📦 Epoch sandbox loaded {sample_count} sample(s)")
        return torch.tensor(loaded, dtype=torch.float32)

# === Train GAN ===
def train_gan():
    G = Generator()
    D = Discriminator()

    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(EPOCHS):
        real = generate_sandboxed_ccd_vectors()

        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        fake = G(z).detach()

        d_real = D(real)
        d_fake = D(fake)

        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        z = torch.randn(BATCH_SIZE, LATENT_DIM)
        fake = G(z)
        d_fake = D(fake)
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        print(f"📊 Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    torch.save(G.state_dict(), MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

# === Run Training ===
if __name__ == "__main__":
    train_gan()
