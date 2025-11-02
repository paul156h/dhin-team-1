# DHIN GAN Project - Synthetic Healthcare Data Generation

A comprehensive toolkit for generating synthetic healthcare data using Generative Adversarial Networks (GANs). 

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
