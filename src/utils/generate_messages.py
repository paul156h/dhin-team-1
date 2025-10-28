import sys
import os
# Add parent directories to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from models.ehr_gan import load_and_encode, load_model, generate_samples, make_adt, make_oru, make_ccd


def main():
    base = os.path.dirname(__file__)
    csv = os.path.abspath(os.path.join(base, '..', '..', 'data', 'datasets', 'mimic-demo-dataset.csv'))
    df, enc, X, inds = load_and_encode(csv)
    # Model is now in src/models directory
    model_path = os.path.abspath(os.path.join(base, '..', 'models', 'simple_gan.pt'))
    if not os.path.exists(model_path):
        print(f'Model not found at {model_path}')
        print('Run train_gan.py first: python3 src/models/train_gan.py')
        return
    G, sizes, noise_dim = load_model(model_path, enc)

    # Example: generate 10 female patients age 50-64
    n = 10
    conditions = []
    for i in range(n):
        conditions.append({'gender': 'F', 'age_range': '50-64'})

    samples = generate_samples(G, sizes, enc, n, conditions=conditions)
    # Output to project outputs directory
    outdir = os.path.abspath(os.path.join(base, '..', '..', 'outputs'))
    os.makedirs(outdir, exist_ok=True)
    for i, p in enumerate(samples):
        pid = f"SYN-{i+1:04d}"
        adt = make_adt(pid, p.get('name', 'Synthetic^Patient'), p.get('gender','U'), p.get('age_range',''), p.get('race',''), p.get('ethnicity',''))
        oru = make_oru(pid)
        ccd = make_ccd({**p, 'patient_id': pid, 'name': 'Synthetic Patient'})
        with open(os.path.join(outdir, f"patient_{i+1:04d}.adt.txt"), 'w', encoding='utf-8') as f:
            f.write(adt)
        with open(os.path.join(outdir, f"patient_{i+1:04d}.oru.txt"), 'w', encoding='utf-8') as f:
            f.write(oru)
        with open(os.path.join(outdir, f"patient_{i+1:04d}.ccd.xml"), 'w', encoding='utf-8') as f:
            f.write(ccd)
    print('Wrote', n, 'patients to', outdir)


if __name__ == '__main__':
    main()
