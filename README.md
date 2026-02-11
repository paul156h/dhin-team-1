# DHIN Project - ADT Message Evaluation Metrics

Quality evaluation system for HL7 ADT messages with scoring and visualization.

## Repository Structure

```
├── evaluation_metrics/     # Evaluation metric scripts
│   ├── adt_evaluation_metrics.py
│   ├── adt_delaware_comparison.py
│   └── create_quality_pdf.py
├── test_data/             # Test ADT messages and sample outputs
│   ├── adt_messages/      # Sample HL7 ADT messages (10 patients)
│   └── sample_outputs/    # Example evaluation reports
├── docs/                  # Documentation
│   └── Evaluation Metrics.md
└── requirements.txt       # Python dependencies
```

## Quick Start

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
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# Evaluate all ADT messages and generate text report
python evaluation_metrics/adt_evaluation_metrics.py

# Generate PDF quality visualization
python evaluation_metrics/create_quality_pdf.py

# Compare with Delaware demographics
python evaluation_metrics/adt_delaware_comparison.py
```

**Output files** are saved to `test_data/sample_outputs/`:
- `adt_evaluation_report.txt` - Detailed quality scores
- `adt_quality_visualization.pdf` - Visual report
- `adt_delaware_comparison.txt` - Demographic analysis

### Test Data

Sample ADT messages (10 patients) are provided in `test_data/adt_messages/` for testing. Example outputs are in `test_data/sample_outputs/`.

## Documentation

See `docs/Evaluation Metrics.md` for detailed information about:
- Scoring methodology
- Required vs standard segments
- Field completeness calculation
- Quality tiers and ratings

---

Created by Jason Martinez, University of Delaware
