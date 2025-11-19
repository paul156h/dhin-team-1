import os
import tempfile
import shutil
import requests
import zipfile
import csv
import random
import torch
import torch.nn as nn

# === CONFIG ===
URL = "https://physionet.org/static/published-projects/mimic-iv-demo/mimic-iv-demo-2.2.zip"
EPOCHS = 50
BATCH_SIZE = 8
SEQ_LEN = 50
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === SANDBOX SETUP ===
sandbox = tempfile.mkdtemp(prefix="sandbox_mimic_")
zip_path = os.path.join(sandbox, "mimic.zip")
print(f"[+] Sandbox created at: {sandbox}")

# === DOWNLOAD AND EXTRACT DATASET ===
print("[+] Downloading MIMIC-IV Demo dataset...")
response = requests.get(URL, stream=True)
if response.status_code != 200:
    print("[-] Failed to download dataset.")
    shutil.rmtree(sandbox)
    exit()

with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("[+] Extracting dataset inside sandbox...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(sandbox)

# === TOKENIZATION ===
def tokenize_claim(row, vocab_map):
    tokens = []
    for field in row:
        field = field.strip()
        if not field:
            continue
        token = vocab_map.setdefault(field, len(vocab_map) + 4)
        tokens.append(token)
    if len(tokens) > SEQ_LEN:
        tokens = tokens[:SEQ_LEN]
    else:
        tokens += [0] * (SEQ_LEN - len(tokens))
    return torch.tensor(tokens)

vocab_map = {}
claim_data = []

print("[+] Parsing CSV files...")
for root, _, files in os.walk(sandbox):
    for filename in files:
        if filename.endswith(".csv"):
            csv_path = os.path.join(root, filename)
            try:
                with open(csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    headers = next(reader, None)
                    for row in reader:
                        claim_data.append(tokenize_claim(row, vocab_map))
            except Exception as e:
                print(f"[-] Skipped {filename}: {e}")
        if len(claim_data) > 5000:
            break
    if len(claim_data) > 5000:
        break

if not claim_data:
    print("[-] No usable claims found. Exiting.")
    shutil.rmtree(sandbox)
    exit()

vocab_size = max(vocab_map.values()) + 1
print(f"[+] Parsed {len(claim_data)} samples, vocab size = {vocab_size}")

# === MODEL DEFINITIONS ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        decoder_layer = nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, z):
        embedded = self.embedding(z)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(SEQ_LEN).to(device)
        output = self.transformer(embedded, embedded, tgt_mask=tgt_mask)
        return self.fc(output)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer(embedded)
        return torch.sigmoid(self.fc(output.mean(dim=0)))

# === TRAINING ===
G = Generator().to(device)
D = Discriminator().to(device)
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)

total_steps = EPOCHS * BATCH_SIZE
progress_interval = total_steps // 10  # 10% increments
step_counter = 0
next_progress_mark = progress_interval

print("[+] Starting training...")
for epoch in range(1, EPOCHS + 1):
    for _ in range(BATCH_SIZE):
        step_counter += 1

        real = random.choice(claim_data).to(device)
        noise = torch.randint(0, vocab_size, (SEQ_LEN,)).to(device)
        fake = torch.argmax(G(noise.unsqueeze(1)), dim=-1).squeeze(1).detach()

        # Train Discriminator
        d_real = D(real)
        d_fake = D(fake)
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        noise = torch.randint(0, vocab_size, (SEQ_LEN,)).to(device)
        fake = G(noise.unsqueeze(1))
        d_fake = D(torch.argmax(fake, dim=-1).squeeze(1))
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Output at every 10% progress
        if step_counter >= next_progress_mark or step_counter == total_steps:
            progress_pct = (step_counter / total_steps) * 100
            with torch.no_grad():
                z = torch.randint(0, vocab_size, (SEQ_LEN,)).to(device)
                generated = torch.argmax(G(z.unsqueeze(1)), dim=-1).squeeze(1)
                tokens = [k for k, v in vocab_map.items() if v in generated.tolist()]
                random_sample = " | ".join(random.sample(tokens, min(len(tokens), 20)))
                print(f"\n[{progress_pct:.0f}%] Randomly Generated Synthetic Record:\n{random_sample}")
            next_progress_mark += progress_interval

# === CLEANUP ===
print("\n[+] Training complete. Cleaning up sandbox...")
shutil.rmtree(sandbox)
print("[+] Sandbox deleted. All temporary data erased.")
