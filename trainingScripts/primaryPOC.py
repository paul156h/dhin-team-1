import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 🔧 USER INPUTS
RAW_MODEL_NAME = "localDemoPOC"
LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "trainingDataset.csv")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = SCRIPT_DIR

# ⚙️ Training parameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

# -------------------------------
# Utility: sanitize model name
# -------------------------------
def sanitize_model_name(name):
    name = re.sub(r'[^a-zA-Z0-9]', '', name)
    if not name or not name[0].isalpha():
        raise ValueError("Model name must start with a letter and contain only letters and numbers.")
    return name

# -------------------------------
# Dataset Loader Template
# -------------------------------
def load_local_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at {path}")
    df = pd.read_csv(path)

    if "TEXT" in df.columns:
        corpus = df["TEXT"].dropna().apply(lambda x: str(x).strip()).tolist()
        source = "TEXT column"
    else:
        corpus = []
        for _, row in df.iterrows():
            note = ",".join(str(row[col]) for col in df.columns)
            corpus.append(note)
        source = "synthetic structured fields"
    return corpus, df, source

class CSVDataset(Dataset):
    def __init__(self, corpus, vocab, max_length=MAX_LENGTH):
        self.corpus = corpus
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        ids = [self.stoi.get(ch, 0) for ch in text]
        ids = ids[:self.max_length]
        ids += [0] * (self.max_length - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

# -------------------------------
# Model Template
# -------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, max_length=MAX_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        out = self.transformer(src_emb.transpose(0,1), tgt_emb.transpose(0,1))
        out = self.fc_out(out).transpose(0,1)
        return out

# -------------------------------
# Training Loop Template
# -------------------------------
def train_model(corpus, output_name, source_info, model_class=SimpleTransformer):
    if not corpus:
        raise ValueError("Corpus is empty — check your CSV file.")

    # Build vocab
    vocab = sorted(set("".join(corpus)))
    vocab_size = len(vocab)

    dataset = CSVDataset(corpus, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        losses = []
        print(f"\n🔄 Starting Epoch {epoch+1}/{EPOCHS}")
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch"):
            x, y = x.to(device), y.to(device)
            output = model(x, x)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1}: Mean Loss = {mean_loss:.4f}")

    # Save standardized object
    save_obj = {
        "state_dict": model.state_dict(),
        "config": {
            "parameters": {
                "sex_assigned_at_birth": {"male": 0.5, "female": 0.5},
                "smoker_status": {"yes": 0.4, "no": 0.6}
            },
            "training_info": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "learning_rate": LR,
                "corpus_source": source_info,
                "num_samples": len(corpus),
                "vocab_size": vocab_size
            },
            "model_class": model_class.__name__,
            "model_hparams": {
                "d_model": 128,
                "nhead": 8,
                "num_layers": 4,
                "max_length": MAX_LENGTH
            },
            "vocab": vocab
        }
    }

    model_path = os.path.join(MODELS_DIR, f"{output_name}.pth")
    torch.save(save_obj, model_path)
    print(f"\n✅ Model + config saved to {model_path}")

# -------------------------------
# Main
# -------------------------------
def main():
    print("🚀 Starting training script...")
    output_name = sanitize_model_name(RAW_MODEL_NAME)
    corpus, df, source_info = load_local_csv(LOCAL_CSV_PATH)
    print(f"Loaded corpus with {len(corpus)} samples from {source_info}")
    train_model(corpus, output_name, source_info, model_class=SimpleTransformer)
    print("✅ Training complete")

if __name__ == "__main__":
    main()
