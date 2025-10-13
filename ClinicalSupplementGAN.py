import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import io
import os

# === Configuration ===
PROJECT = "TCGA-LUAD"  # Change to any TCGA project (e.g., TCGA-LUAD, TCGA-COAD)
MAX_FILES = 15         # Number of clinical supplement files to stream
MODEL_NAME = f"gan_{PROJECT.lower().replace('-', '_')}_clinical.pth"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_NAME)

# === GAN Architecture ===
class Generator(nn.Module):
    def __init__(self, z_dim=10, output_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# === Stream Clinical Supplement Data ===
def fetch_clinical_supplements(project=PROJECT, max_files=MAX_FILES):
    print(f"🔄 Streaming clinical supplement files for {project}...")
    query_url = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "data_category", "value": "Clinical"}},
            {"op": "=", "content": {"field": "data_type", "value": "Clinical Supplement"}},
            {"op": "=", "content": {"field": "project_id", "value": project}}
        ]
    }

    params = {
        "filters": str(filters).replace("'", '"'),
        "fields": "file_id",
        "format": "JSON",
        "size": max_files
    }

    response = requests.get(query_url, params=params)
    file_ids = [f["file_id"] for f in response.json().get("data", {}).get("hits", [])]

    dfs = []
    for file_id in file_ids:
        data_url = f"https://api.gdc.cancer.gov/data/{file_id}"
        file_response = requests.get(data_url)
        try:
            df = pd.read_csv(io.StringIO(file_response.content.decode("utf-8")), sep="\t")
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipped file {file_id}: {e}")

    if not dfs:
        raise ValueError("❌ No clinical supplement files could be parsed. Try a different project or increase max_files.")

    print(f"✅ Parsed {len(dfs)} clinical files.")
    return pd.concat(dfs, ignore_index=True)

# === Standardize Clinical Data ===
def standardize_clinical_data(df):
    possible_columns = {
        "age": ["age_at_diagnosis", "age_at_initial_pathologic_diagnosis"],
        "gender": ["gender"],
        "stage": ["tumor_stage", "pathologic_stage"],
        "treatment": ["treatment_type", "therapy_type"]
    }

    def find_column(df, options):
        for col in options:
            if col in df.columns:
                return col
        return None

    age_col = find_column(df, possible_columns["age"])
    gender_col = find_column(df, possible_columns["gender"])
    stage_col = find_column(df, possible_columns["stage"])
    treatment_col = find_column(df, possible_columns["treatment"])

    if not all([age_col, gender_col, stage_col, treatment_col]):
        raise ValueError("❌ Required clinical columns not found in streamed data.")

    df = df[[age_col, gender_col, stage_col, treatment_col]].dropna()

    df["gender"] = df[gender_col].str.lower().map({"male": 0, "female": 1})
    df["stage"] = df[stage_col].str.extract(r'(\d+)').astype(float)
    df["age"] = (df[age_col] - df[age_col].min()) / (df[age_col].max() - df[age_col].min())

    df["treatment"] = df[treatment_col].astype(str).str.lower()
    treatment_map = {t: i for i, t in enumerate(df["treatment"].unique())}
    df["treatment"] = df["treatment"].map(treatment_map)
    df["treatment"] = df["treatment"] / df["treatment"].max()

    print(f"✅ Standardized {len(df)} clinical records.")
    return df[["age", "gender", "stage", "treatment"]]

# === Train GAN ===
def train_gan(df, epochs=500, batch_size=32):
    z_dim = 10
    output_dim = df.shape[1]

    G = Generator(z_dim, output_dim)
    D = Discriminator(output_dim)

    criterion = nn.BCELoss()
    g_opt = torch.optim.Adam(G.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(D.parameters(), lr=0.0002)

    data = torch.tensor(df.values, dtype=torch.float32)

    for epoch in range(epochs):
        idx = np.random.randint(0, data.size(0), batch_size)
        real = data[idx]

        z = torch.randn(batch_size, z_dim)
        fake = G(z).detach()

        d_real = D(real)
        d_fake = D(fake)

        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        z = torch.randn(batch_size, z_dim)
        fake = G(z)
        d_fake = D(fake)
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        if epoch % 50 == 0:
            print(f"📊 Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

    torch.save(G.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# === Run Training ===
if __name__ == "__main__":
    try:
        raw_df = fetch_clinical_supplements()
        clean_df = standardize_clinical_data(raw_df)
        train_gan(clean_df)
    except Exception as e:
        print(f"❌ Error: {e}")
