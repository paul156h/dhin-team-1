# DHIN GAN Project - Synthetic Healthcare Data Generation

A comprehensive toolkit for generating synthetic healthcare data using Generative Adversarial Networks (GANs). This project implements various GAN architectures to create realistic synthetic clinical data while preserving privacy and statistical properties of real healthcare datasets.

## 🏥 Project Overview

This project focuses on generating synthetic healthcare data for research and development purposes while maintaining patient privacy. It includes multiple GAN implementations tailored for different types of clinical data:

- **Clinical Supplement Data**: TCGA cancer clinical data generation
- **EHR Data**: Electronic Health Records synthesis
- **Claims Data**: Healthcare claims and billing data generation
- **General Medical Data**: Flexible GAN for various healthcare datasets

## 📁 Repository Structure

```
dhin-team-1/
├── src/                          # Source code
│   ├── models/                   # GAN model implementations
│   │   ├── ClinicalSupplementGAN.py
│   │   ├── ehr_gan.py
│   │   ├── ccdGAN.py
│   │   ├── genericGAN.py
│   │   └── train_gan.py
│   ├── gui/                      # GUI applications
│   │   ├── TKinterWindow.py      # Main GAN selector interface
│   │   └── TestMenu.py           # GAN training visualizer
│   └── utils/                    # Utility scripts
│       ├── claimsPOC.py
│       └── generate_messages.py
├── data/                         # Data storage
│   └── datasets/                 # Dataset files
│       └── mimic-demo-dataset.csv
├── models/                       # Trained models
│   └── trained/                  # Pre-trained model files
│       ├── MnistGAN.pth
│       └── ccdGAN.pth
├── docs/                         # Documentation
│   ├── README_GAN.md
│   ├── resources.md
│   └── project_notes.txt
├── examples/                     # Usage examples
├── scripts/                      # Utility scripts
├── outputs/                      # Generated outputs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/paul156h/dhin-team-1.git
   cd dhin-team-1
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv gan-env
   source gan-env/bin/activate  # On Windows: gan-env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies (Linux)**
   ```bash
   # For GUI support
   sudo apt install python3-tk  # Ubuntu/Debian
   sudo dnf install python3-tkinter  # Fedora
   ```

### Usage

#### GUI Applications

1. **GAN Selector Interface**
   ```bash
   python src/gui/TKinterWindow.py
   ```
   
2. **Training Visualizer**
   ```bash
   python src/gui/TestMenu.py
   ```

#### CLI Model Training

1. **Clinical Supplement GAN**
   ```bash
   python src/models/ClinicalSupplementGAN.py
   ```

2. **EHR GAN**
   ```bash
   python src/models/ehr_gan.py
   ```

3. **Generic GAN**
   ```bash
   python src/models/genericGAN.py
   ```

## 🧠 Available Models

### ClinicalSupplementGAN
- **Purpose**: Generate synthetic TCGA clinical supplement data
- **Data Source**: Cancer clinical data from TCGA database
- **Features**: Automated data fetching, preprocessing, and GAN training
- **Output**: Synthetic patient clinical records

### EHR GAN
- **Purpose**: Electronic Health Records synthesis
- **Features**: Patient timeline generation, medical code sequences
- **Use Cases**: EHR system testing, research data augmentation

### Generic GAN
- **Purpose**: Flexible GAN for custom healthcare datasets
- **Features**: Configurable architecture, multiple data formats
- **Use Cases**: Custom medical data generation

## 📊 Datasets

The project works with various healthcare datasets:

- **TCGA Clinical Data**: Cancer patient clinical supplements
- **MIMIC Demo Dataset**: Synthetic critical care data
- **Custom Clinical Data**: User-provided healthcare datasets

## 🔧 Configuration

Model configurations can be adjusted in individual Python files or through the GUI interface. Key parameters include:

- Learning rates
- Batch sizes
- Network architectures
- Training epochs
- Data preprocessing options

## 📋 Requirements

See `requirements.txt` for complete dependency list. Core dependencies include:

- torch >= 1.9.0
- torchvision >= 0.10.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- tkinter (system package)
- requests >= 2.25.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

DHIN Team 1 - Digital Health Innovation Network

## 📚 Additional Documentation

- [GAN Model Documentation](docs/README_GAN.md)
- [Resource Links](docs/resources.md)
- [Project Notes](docs/project_notes.txt)

## ⚠️ Important Notes

- This project generates synthetic data for research purposes
- Ensure compliance with healthcare data regulations (HIPAA, GDPR)
- Synthetic data should be validated before use in production
- Always follow institutional review board guidelines for research use
Creating synthetic data pool for claims, HL7 results and care summaries (consolidated and encounter).
