#!/usr/bin/env python3
"""
Compare ADT message demographics to Delaware census percentages.
Outputs race, ethnicity, gender, and age group distributions.
"""

from pathlib import Path
from collections import Counter

# Delaware 2020 Census demographics (example percentages)
DELAWARE_CONSENSUS = {
    'race': {
        'White': 61.6,
        'Black': 12.4,
        'Asian': 6.0,
        'Hispanic': 18.7,
        'American Indian': 1.1,
        'Pacific Islander': 0.2,
        'Other': 8.4,
        'Two or More': 10.2
    },
    'gender': {
        'M': 48.5,
        'F': 51.5
    },
    'ethnicity': {
        'Hispanic or Latino': 18.7,
        'Not Hispanic or Latino': 81.3
    },
    'age_group': {
        '<18': 21.7,
        '18-24': 8.9,
        '25-44': 25.6,
        '45-64': 27.2,
        '65+': 16.6
    }
}

# PID segment field indices (0-based after split by |)
# PID|1|ID|ALT-ID|NAME||DOB|Gender|...
# 0   1  2  3      4   5 6   7      8
PID_GENDER_IDX = 7
PID_DOB_IDX = 6
PID_RACE_IDX = 10
PID_ETHNICITY_IDX = 22


def parse_pid_line(pid_line):
    """Extract demographic fields from PID segment."""
    fields = pid_line.strip().split('|')
    
    gender = fields[PID_GENDER_IDX] if len(fields) > PID_GENDER_IDX else ''
    dob = fields[PID_DOB_IDX] if len(fields) > PID_DOB_IDX else ''
    race = fields[PID_RACE_IDX] if len(fields) > PID_RACE_IDX else ''
    ethnicity = fields[PID_ETHNICITY_IDX] if len(fields) > PID_ETHNICITY_IDX else ''
    
    return {
        'gender': gender.strip() or 'Unknown',
        'dob': dob.strip(),
        'race': race.strip() or 'Unknown',
        'ethnicity': ethnicity.strip() or 'Unknown'
    }


def get_age_group(dob):
    """Convert DOB (YYYYMMDD format) to age group."""
    if not dob or len(dob) < 4:
        return 'Unknown'
    
    try:
        birth_year = int(dob[:4])
        age = 2025 - birth_year
        
        if age < 18:
            return '<18'
        elif age < 25:
            return '18-24'
        elif age < 45:
            return '25-44'
        elif age < 65:
            return '45-64'
        else:
            return '65+'
    except (ValueError, IndexError):
        return 'Unknown'


def calculate_percentages(items):
    """Calculate percentage distribution of items."""
    if not items:
        return {}
    
    counter = Counter(items)
    total = len(items)
    
    return {k: round(v / total * 100, 1) for k, v in counter.items()}


def similarity_score(adt_dist, delaware_dist, threshold=5.0):
    """
    Calculate how similar ADT distribution is to Delaware.
    Returns True if all categories match within threshold percentage points.
    """
    matches = 0
    total_categories = len(delaware_dist)
    
    for category, delaware_pct in delaware_dist.items():
        adt_pct = adt_dist.get(category, 0)
        if abs(adt_pct - delaware_pct) <= threshold:
            matches += 1
    
    return matches == total_categories


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    adt_dir = project_root / "data" / "adt_messages"
    output_dir = project_root / "outputs"
    
    if not adt_dir.exists():
        print(f"Error: ADT directory not found: {adt_dir}")
        return
    
    # Collect demographics from all ADT messages
    genders = []
    races = []
    ethnicities = []
    age_groups = []
    
    adt_files = list(adt_dir.glob('*.adt.txt'))
    print(f"Processing {len(adt_files)} ADT messages...")
    
    for adt_file in adt_files:
        with open(adt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('PID|'):
                    demographics = parse_pid_line(line)
                    genders.append(demographics['gender'])
                    races.append(demographics['race'])
                    ethnicities.append(demographics['ethnicity'])
                    age_groups.append(get_age_group(demographics['dob']))
                    break  # Only first PID per message
    
    # Calculate percentages
    gender_pcts = calculate_percentages(genders)
    race_pcts = calculate_percentages(races)
    ethnicity_pcts = calculate_percentages(ethnicities)
    age_pcts = calculate_percentages(age_groups)
    
    # Generate report
    output_file = output_dir / "adt_delaware_comparison.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ADT MESSAGE DEMOGRAPHICS vs DELAWARE CENSUS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total ADT Messages Analyzed: {len(adt_files)}\n")
        f.write(f"Total Patients: {len(genders)}\n\n")
        
        # Gender Comparison
        f.write("GENDER DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        for gender, pct in sorted(gender_pcts.items()):
            f.write(f"  {gender:20s}: {pct:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for gender, pct in sorted(DELAWARE_CONSENSUS['gender'].items()):
            f.write(f"  {gender:20s}: {pct:6.1f}%\n")
        similar = similarity_score(gender_pcts, DELAWARE_CONSENSUS['gender'])
        f.write(f"\nWithin ±5% threshold: {'YES ✓' if similar else 'NO ✗'}\n\n")
    
        
        # Age Group Comparison
        f.write("AGE GROUP DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        age_order = ['<18', '18-24', '25-44', '45-64', '65+', 'Unknown']
        for age_group in age_order:
            if age_group in age_pcts:
                f.write(f"  {age_group:20s}: {age_pcts[age_group]:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for age_group in age_order:
            if age_group in DELAWARE_CONSENSUS['age_group']:
                f.write(f"  {age_group:20s}: {DELAWARE_CONSENSUS['age_group'][age_group]:6.1f}%\n")
        similar = similarity_score(age_pcts, DELAWARE_CONSENSUS['age_group'])
        f.write(f"\nWithin ±5% threshold: {'YES ✓' if similar else 'NO ✗'}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Note: Threshold of ±5% is used for similarity comparison.\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Comparison report saved to: {output_file}")


if __name__ == "__main__":
    main()
