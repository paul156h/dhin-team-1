# Training hyperparameters and CSV configuration.
# Adjust these to control model size, training duration, and input format.

import os, csv, torch, math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# -------------------------------
# Parameters
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "trainingDataset.csv")
HAS_HEADERS = True
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-3
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
GRAD_ACCUM = 4
USE_AMP = torch.cuda.is_available()
MODEL_OUT = "trained_model.pth"
PAD_TOKEN = "<PAD>"

# -------------------------------
# Metadata scan
# -------------------------------
def scan_csv_for_metadata(csv_path, has_headers=True):
    # Scan the CSV to infer metadata needed for training and GUI features.
    # Extracts vocab, max row length, headers, and binary column candidates.
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = None
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV appears to be empty.")

        if has_headers:
            headers = first_row
        else:
            f.seek(0)
            reader = csv.reader(f)

        col_count = len(headers) if headers else None
        binary_candidates = [set() for _ in range(col_count)] if headers else None

        vocab_chars = set()
        max_len = 0
        num_rows = 0

        for row in reader:
            # Track longest row and collect all characters for vocabulary construction.
            if not any(cell.strip() for cell in row):
                continue
            num_rows += 1
            line = ",".join(row)
            max_len = max(max_len, len(line) + 1)
            vocab_chars.update(line)

            if headers:
                for i, cell in enumerate(row):
                    val = cell.strip()
                    if val != "":
                        if len(binary_candidates[i]) <= 3:
                            binary_candidates[i].add(val)

        # Finalize binary columns
        # Identify columns with exactly two unique values (used for GUI sliders).
        binary_columns = {}
        if headers and binary_candidates:
            for i, vals in enumerate(binary_candidates):
                cleaned = [v for v in vals if v != ""]
                if len(cleaned) == 2:
                    binary_columns[headers[i]] = cleaned

        vocab = sorted(vocab_chars)
        if "\n" not in vocab:
            vocab.append("\n")

        return {
            "headers": headers,
            "has_headers": has_headers,
            "binary_columns": binary_columns,
            "vocab": [PAD_TOKEN] + vocab,
            "max_length": max_len,
            "num_rows": num_rows
        }

# Iterable dataset that streams CSV rows and converts them into
# fixed-length character ID sequences for next-character prediction.
# -------------------------------
# Dataset
# -------------------------------
class CSVRowIterableDataset(IterableDataset):
    def __init__(self, csv_path, vocab, max_length, has_headers=True):
        super().__init__()
        self.csv_path = csv_path
        self.has_headers = has_headers
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.pad_id = 0
        self.max_length = max_length

    def __iter__(self):
        def encode_line(line):
            # Convert a CSV row into (x, y) where y is x shifted by one character.
            ids = [self.stoi.get(ch, self.pad_id) for ch in line]
            ids = ids[:self.max_length] + [self.pad_id] * max(0, self.max_length - len(ids))
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            return x, y

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if self.has_headers:
                next(reader, None)  # skip header row
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                line = ",".join(row) + "\n"
                yield encode_line(line)

# -------------------------------
# Model
# -------------------------------
class CharTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_length, pad_id=0, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_id = pad_id

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        pad_mask = x.eq(self.pad_id)
        out = self.transformer(emb, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.fc_out(out)

# Main training loop: builds dataset, trains the Transformer,
# and saves model weights + metadata for use in the GUI.
# -------------------------------
# Training loop
# -------------------------------
def train(csv_path, has_headers=True):
    if not csv_path:
        raise ValueError("CSV_PATH is empty. Set CSV_PATH to your training CSV file.")
    if not csv_path.lower().endswith(".csv"):
        raise ValueError(f"CSV_PATH must point to a .csv file, got: {csv_path}")

    meta = scan_csv_for_metadata(csv_path, has_headers)
    vocab, max_len, num_rows = meta["vocab"], meta["max_length"], meta["num_rows"]
    if max_len < 2:
        max_len = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = 0

    model = CharTransformerDecoder(
        vocab_size=len(vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        max_length=max_len,
        pad_id=pad_id,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    dataset = CSVRowIterableDataset(csv_path, vocab, max_len, has_headers)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        # Gradient accumulation allows effective larger batch sizes.
        running_loss = 0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (x,y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            x,y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits.reshape(-1,len(vocab)), y.reshape(-1))
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (step+1) % GRAD_ACCUM == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()*GRAD_ACCUM
            steps += 1

        if steps % GRAD_ACCUM != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = running_loss / max(1, steps)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

    # Save model + minimal config
    # Save only the state_dict and essential metadata needed to rebuild the model.
    save_obj = {
        "state_dict": model.state_dict(),
        "config": {
            "format_version": 2,
            "model_type": "char_transformer_decoder",
            "model_hparams": {
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "max_length": max_len,
                "dropout": DROPOUT,
                "pad_id": pad_id,
            },
            "headers": meta["headers"],
            "has_headers": meta["has_headers"],
            "binary_columns": meta["binary_columns"],
            "vocab": vocab,
            "max_length": max_len,
            "pad_id": pad_id,
            "num_rows": num_rows,
        },
    }
    torch.save(save_obj, MODEL_OUT)
    print(f"✅ Model + config saved to {MODEL_OUT}")

# -------------------------------
if __name__ == "__main__":
    train(CSV_PATH, HAS_HEADERS)
