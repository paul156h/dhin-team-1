import sys
import os
# Add parent directories to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from models.ehr_gan import load_and_encode, load_model, generate_samples, make_adt, make_oru, make_ccd


def get_bundled_dir():
    """Get the directory where bundled files are located when running as PyInstaller exe."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running as normal Python script
        return os.path.dirname(__file__)


def main():
    base = get_bundled_dir()
    
    # Determine paths based on whether we're bundled or not
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable - data files are in _internal
        csv = os.path.join(base, '_internal', 'data', 'MIMIC', 'mimic-demo-dataset.csv')
        model_path = os.path.join(base, '_internal', 'models', 'simple_gan.pt')
    else:
        # Running as normal Python script
        csv = os.path.abspath(os.path.join(base, '..', '..', 'data', 'MIMIC', 'mimic-demo-dataset.csv'))
        model_path = os.path.abspath(os.path.join(base, '..', 'models', 'simple_gan.pt'))
    
    if not os.path.exists(csv):
        print(f'CSV file not found at {csv}')
        return
    
    df, enc, X, inds = load_and_encode(csv)
    # Model is now in src/models directory
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
    if getattr(sys, 'frozen', False):
        # Running as executable - put outputs next to the exe
        outdir = os.path.join(os.path.dirname(sys.executable), 'outputs')
    else:
        # Running as script
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
