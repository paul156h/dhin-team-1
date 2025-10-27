import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import re
import tempfile
import shutil
import requests
import zipfile
import csv
import torch
import torch.nn as nn
import random

# === STEP 1: USER-SUPPLIED CONFIGURATION ===
# These values should be filled in by a chatbot or coder
MODEL_TYPE = "Transformer"  # e.g., "Transformer", "GAN", "LSTM"
DOCUMENT_TYPE = "insurance claims"  # e.g., "clinical notes", "recipes"
DATASET_URL = "https://example.com/dataset.zip"  # Replace with actual dataset URL

LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 16
SEQ_LEN = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === STEP 2: DOWNLOAD & EXTRACT DATASET ===
sandbox = tempfile.mkdtemp(prefix="ml_sandbox_")
zip_path = os.path.join(sandbox, "dataset.zip")
print(f"[+] Sandbox created at: {sandbox}")

print("[+] Downloading dataset...")
response = requests.get(DATASET_URL, stream=True)
if response.status_code != 200:
    print("[-] Failed to download dataset.")
    shutil.rmtree(sandbox)
    exit()

with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("[+] Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(sandbox)

# === STEP 3: PARSE DATA & INFER CONSTRAINTS ===
samples = []
field_counts = {}

for root, _, files in os.walk(sandbox):
    for filename in files:
        if filename.endswith(".csv"):
            try:
                with open(os.path.join(root, filename), encoding="utf-8") as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    for row in reader:
                        samples.append(row)
                        for i, val in enumerate(row):
                            key = headers[i] if headers else f"Field {i}"
                            field_counts.setdefault(key, set()).add(val.strip())
            except Exception as e:
                print(f"[-] Skipped {filename}: {e}")
        if len(samples) > 5000:
            break
    if len(samples) > 5000:
        break

if not samples:
    print("[-] No usable data found.")
    shutil.rmtree(sandbox)
    exit()

# === STEP 4: AUTO-GENERATE CONSTRAINTS ===
def get_constraints():
    constraints = {}
    for field, values in field_counts.items():
        if 2 <= len(values) <= 10:
            for val in sorted(values):
                constraints[f"{field}: {val} %"] = 0
    return constraints

# === STEP 5: DEFINE MODEL ===
# === CODER OR CHATBOT MUST MODIFY ===
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 64)  # Placeholder
        self.encoder = nn.Sequential(
            nn.Linear(SEQ_LEN * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1000)
        )

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        return self.encoder(x)

model = CustomModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === STEP 6: TRAINING LOOP ===
print("[+] Starting training...")
total_steps = EPOCHS * BATCH_SIZE
progress_interval = total_steps // 10
step_counter = 0
next_progress_mark = progress_interval

# Fake tokenized data for placeholder training
training_data = [torch.randint(0, 1000, (SEQ_LEN,)) for _ in range(len(samples))]

for epoch in range(EPOCHS):
    for _ in range(BATCH_SIZE):
        step_counter += 1
        sample = random.choice(training_data).unsqueeze(0).to(device)
        target = sample.clone().to(device)

        output = model(sample)
        loss = criterion(output, target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_counter >= next_progress_mark or step_counter == total_steps:
            with torch.no_grad():
                test_input = random.choice(training_data).unsqueeze(0).to(device)
                test_output = model(test_input)
                print(f"\n[{(step_counter / total_steps) * 100:.0f}%] Sample Output:\n{test_output}")
            next_progress_mark += progress_interval

# === STEP 7: CLEANUP ===
shutil.rmtree(sandbox)
print("[+] Training complete. Sandbox deleted.")

# === STEP 8: PROMPT FOR VALID MODEL NAME ===
def get_valid_filename():
    while True:
        name = input("Enter model name (letters and numbers only): ").strip()
        if re.fullmatch(r"[A-Za-z0-9]+", name):
            return name + ".pth"
        print("[-] Invalid name. Use only letters and numbers, no spaces or symbols.")

filename = get_valid_filename()

# === STEP 9: SAVE MODEL ===
# Save to src/models directory
model_path = os.path.join(os.path.dirname(__file__), filename)
torch.save(model.state_dict(), model_path)
print(f"[+] Model saved as: {model_path}")
