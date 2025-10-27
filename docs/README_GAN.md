# EHR GAN demo

This contains a minimal conditional GAN example that trains on `mimic-demo-dataset.csv` to produce synthetic patient demographics and then renders simple HL7 ADT, HL7 ORU, and CCD XML templates.

Requirements
- Python 3.8+
- Install packages from `requirements.txt` (CPU-only PyTorch recommended unless you have CUDA)

Quickstart
1. Train the model:

```powershell
python .\train_gan.py
```

2. Generate sample messages:

```powershell
python .\generate_messages.py
```

Outputs will be written to `outputs/`.

Notes
- This is a minimal educational demo. It is NOT production-ready and does not provide privacy guarantees. For production: consider CTGAN, DP mechanisms, and clinical message conformance libraries.
