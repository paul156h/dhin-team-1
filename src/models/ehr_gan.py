import os
import json
import time
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


class CategoricalEncoder:
    """Encode categorical columns to integer labels and one-hot vectors."""
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.maps: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame):
        for c in self.columns:
            vals = list(df[c].fillna("Unknown").astype(str).unique())
            vals_sorted = sorted(vals)
            self.maps[c] = vals_sorted

    def transform_row(self, row: pd.Series) -> Tuple[np.ndarray, Dict[str,int]]:
        parts = []
        inds = {}
        for c in self.columns:
            vals = self.maps[c]
            v = str(row[c]) if pd.notna(row[c]) else "Unknown"
            try:
                idx = vals.index(v)
            except ValueError:
                # unseen value -> add Unknown index (fallback to 0)
                idx = 0
            onehot = np.zeros(len(vals), dtype=np.float32)
            onehot[idx] = 1.0
            parts.append(onehot)
            inds[c] = idx
        return np.concatenate(parts), inds

    def transform_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str,int]]]:
        rows = []
        inds_list = []
        for _, r in df.iterrows():
            v, inds = self.transform_row(r)
            rows.append(v)
            inds_list.append(inds)
        return np.stack(rows, axis=0), inds_list

    def sizes(self) -> List[int]:
        return [len(self.maps[c]) for c in self.columns]

    def decode(self, indices: Dict[str,int]) -> Dict[str,str]:
        out = {}
        for c, i in indices.items():
            out[c] = self.maps[c][i]
        return out


def load_and_encode(csv_path: str, columns=None):
    df = pd.read_csv(csv_path)
    if columns is None:
        columns = ["gender", "age_range", "race", "ethnicity"]
    enc = CategoricalEncoder(columns)
    enc.fit(df)
    X, inds = enc.transform_df(df)
    return df, enc, X, inds


class Generator(nn.Module):
    def __init__(self, noise_dim: int, cond_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        logits = self.net(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        xc = torch.cat([x, c], dim=1)
        return self.net(xc)


def onehot_segments_from_sizes(sizes: List[int]) -> List[Tuple[int,int]]:
    segs = []
    start = 0
    for s in sizes:
        segs.append((start, start + s))
        start += s
    return segs


def train_gan(X: np.ndarray, enc: CategoricalEncoder, epochs=200, batch_size=64, lr=2e-4, save_path=None):
    device = torch.device("cpu")
    N, D = X.shape
    sizes = enc.sizes()
    segs = onehot_segments_from_sizes(sizes)
    cond_dim = sum(sizes)  # we'll use the same one-hot as condition during training
    noise_dim = 32

    G = Generator(noise_dim, cond_dim, D).to(device)
    Dnet = Discriminator(D, cond_dim).to(device)

    optimG = optim.Adam(G.parameters(), lr=lr)
    optimD = optim.Adam(Dnet.parameters(), lr=lr)
    bce = nn.BCELoss()

    X_tensor = torch.from_numpy(X).float().to(device)

    print(f"Starting training on {N} samples, dim={D}, epochs={epochs}")
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            real = X_tensor[idx]
            cond = real.clone()  # using attributes as condition

            # Train Discriminator
            z = torch.randn(len(idx), noise_dim).to(device)
            fake_logits = G(z, cond)
            # apply per-segment softmax to convert to approx one-hot
            fake = torch.zeros_like(fake_logits)
            for (s,e) in segs:
                seg = fake_logits[:, s:e]
                fake[:, s:e] = torch.softmax(seg, dim=1)

            D_real = Dnet(real, cond)
            D_fake = Dnet(fake.detach(), cond)
            lossD = bce(D_real, torch.ones_like(D_real)) + bce(D_fake, torch.zeros_like(D_fake))
            optimD.zero_grad()
            lossD.backward()
            optimD.step()

            # Train Generator
            D_fake_forG = Dnet(fake, cond)
            lossG = bce(D_fake_forG, torch.ones_like(D_fake_forG))
            optimG.zero_grad()
            lossG.backward()
            optimG.step()

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch+1}/{epochs} lossD={lossD.item():.4f} lossG={lossG.item():.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'G': G.state_dict(),
            'D': Dnet.state_dict(),
            'sizes': sizes,
            'noise_dim': noise_dim,
        }, save_path)
        print(f"Saved model to {save_path}")
    return G, Dnet


def load_model(path: str, enc: CategoricalEncoder):
    device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    sizes = ckpt.get('sizes') or enc.sizes()
    noise_dim = ckpt.get('noise_dim', 32)
    D = sum(sizes)
    G = Generator(noise_dim, D, D)
    G.load_state_dict(ckpt['G'])
    G.eval()
    return G, sizes, noise_dim


def generate_samples(G: Generator, sizes: List[int], enc: CategoricalEncoder, n: int, conditions: List[Dict[str,str]] = None):
    device = torch.device("cpu")
    D = sum(sizes)
    segs = onehot_segments_from_sizes(sizes)
    noise_dim = 32
    G.to(device)
    out = []
    batch = 64
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, noise_dim).to(device)
        if conditions:
            # build cond matrix from provided dicts
            cond_mat = np.zeros((b, D), dtype=np.float32)
            for j in range(b):
                cond = conditions[i + j]
                # map to indices
                for ci, col in enumerate(enc.columns):
                    vals = enc.maps[col]
                    v = cond.get(col, None)
                    if v is None:
                        continue
                    try:
                        idx = vals.index(v)
                    except ValueError:
                        idx = 0
                    s,e = segs[ci]
                    cond_mat[j, s+idx] = 1.0
            c = torch.from_numpy(cond_mat).float().to(device)
        else:
            # sample random conditions from encoder empirical distribution
            # default: uniformly random one-hot per segment
            cond_mat = np.zeros((b, D), dtype=np.float32)
            for j in range(b):
                for si, (s,e) in enumerate(segs):
                    k = np.random.randint(e-s)
                    cond_mat[j, s+k] = 1.0
            c = torch.from_numpy(cond_mat).float().to(device)

        logits = G(z, c)
        logits = logits.detach().cpu().numpy()
        # per-segment softmax and argmax to discrete categories
        samples = []
        for row in logits:
            sample_inds = {}
            for si, (s,e) in enumerate(segs):
                seg = row[s:e]
                if len(seg) == 0:
                    continue
                idx = int(np.argmax(seg))
                sample_inds[enc.columns[si]] = idx
            samples.append(enc.decode(sample_inds))
        out.extend(samples)
    return out


# Simple HL7/CCD templates
def make_msh(sending_app='EHRGAN', receiving_app='EHRReceiver', message_type='ADT^A01'):
    ts = time.strftime('%Y%m%d%H%M%S')
    control_id = f"{int(time.time())}"
    return f"MSH|^~\\&|{sending_app}|{sending_app}Fac|{receiving_app}|{receiving_app}Fac|{ts}||{message_type}|{control_id}|P|2.5|||||NE|AL|USA||||\r"


def make_pid(patient_id: str, name='Synthetic^Patient', gender='U', birth_year='1970'):
    # PID|1|patient_id||name||DOB|gender
    dob = f"{birth_year}0101"
    return f"PID|1|{patient_id}||{name}||{dob}|{gender}\r"


def make_adt(patient_id: str, name, gender, age_range, race, ethnicity):
    msh = make_msh(message_type='ADT^A01')
    # EVN segment required for ADT
    evn = f"EVN|A01|{time.strftime('%Y%m%d%H%M%S')}|||||\r"
    # approximate birth year from age_range
    birth_year = '1970'
    pid = make_pid(patient_id, name, gender=gender, birth_year=birth_year)
    # Add race and ethnicity to PD1
    pd1 = f"PD1||||{race}|{ethnicity}||||||||\r"
    # More complete PV1 for admission
    pv1 = "PV1|1|I|WARD^123^1|E|||123^ATTENDING^DOC^A|||MED||||1|||123^REFERRING^DOC^A|IP||VN|||||||||||||||||01|||||||{time.strftime('%Y%m%d%H%M%S')}|\r"
    return msh + evn + pid + pd1 + pv1


def make_oru(patient_id: str, observation='Hemoglobin', value='13.2', units='g/dL'):
    msh = make_msh(sending_app='EHRGAN', receiving_app='LabSystem', message_type='ORU^R01')
    pid = make_pid(patient_id)
    # More complete OBR with required fields
    ts = time.strftime('%Y%m%d%H%M%S')
    obr = f"OBR|1|LAB{patient_id}|ACC{patient_id}|{observation}|||{ts}|{ts}||||||{ts}||123^ORDERING^DOC|||||||F||^^^{ts}|||||||||||\r"
    # OBX with reference range and more complete fields
    obx = f"OBX|1|NM|{observation}^{observation} Test||{value}|{units}|12.0-16.0|N|||F|||{ts}|LAB1\r"
    # Add interpretive notes
    nte = "NTE|1||Sample lab result for demonstration\r"
    return msh + pid + obr + obx + nte


def make_ccd(patient_dict: Dict[str,str]):
    # CCD/CDA R2 compliant template
    pid = patient_dict.get('patient_id', 'SYN-1')
    name = patient_dict.get('name', 'Synthetic Patient')
    gender = patient_dict.get('gender', 'U')
    age_range = patient_dict.get('age_range', '')
    race = patient_dict.get('race', '')
    ethnicity = patient_dict.get('ethnicity', '')
    current_time = time.strftime('%Y%m%d%H%M%S')
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="CDA.xsl"?>
<ClinicalDocument xmlns="urn:hl7-org:v3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <typeId root="2.16.840.1.113883.1.3" extension="POCD_HD000040"/>
    <templateId root="2.16.840.1.113883.10.20.22.1.1"/> <!-- US Realm Header -->
    <templateId root="2.16.840.1.113883.10.20.22.1.2"/> <!-- Continuity of Care Document -->
    <id root="2.16.840.1.113883.19.5.99999.1" extension="{pid}"/>
    <code code="34133-9" codeSystem="2.16.840.1.113883.6.1" codeSystemName="LOINC" displayName="Summarization of Episode Note"/>
    <title>Continuity of Care Document</title>
    <effectiveTime value="{current_time}"/>
    <confidentialityCode code="N" codeSystem="2.16.840.1.113883.5.25"/>
    <languageCode code="en-US"/>
    <recordTarget>
        <patientRole>
            <id extension="{pid}" root="2.16.840.1.113883.19.5.99999.2"/>
            <patient>
                <name>
                    <given>{name.split('^')[0]}</given>
                    <family>{name.split('^')[1] if '^' in name else ''}</family>
                </name>
                <administrativeGenderCode code="{gender}" codeSystem="2.16.840.1.113883.5.1"/>
                <raceCode code="{race}" codeSystem="2.16.840.1.113883.6.238"/>
                <ethnicGroupCode code="{ethnicity}" codeSystem="2.16.840.1.113883.6.238"/>
            </patient>
        </patientRole>
    </recordTarget>
    <component>
        <structuredBody>
            <!-- Required CCD sections -->
            <component>
                <section>
                    <templateId root="2.16.840.1.113883.10.20.22.2.6.1"/> <!-- Allergies -->
                    <code code="48765-2" codeSystem="2.16.840.1.113883.6.1"/>
                    <title>Allergies, Adverse Reactions, and Alerts</title>
                    <text>No known allergies</text>
                </section>
            </component>
            <component>
                <section>
                    <templateId root="2.16.840.1.113883.10.20.22.2.1.1"/> <!-- Medications -->
                    <code code="10160-0" codeSystem="2.16.840.1.113883.6.1"/>
                    <title>Medications</title>
                    <text>No active medications</text>
                </section>
            </component>
            <component>
                <section>
                    <templateId root="2.16.840.1.113883.10.20.22.2.5.1"/> <!-- Problems -->
                    <code code="11450-4" codeSystem="2.16.840.1.113883.6.1"/>
                    <title>Problems</title>
                    <text>No active problems</text>
                </section>
            </component>
        </structuredBody>
    </component>
</ClinicalDocument>
"""
    return xml
