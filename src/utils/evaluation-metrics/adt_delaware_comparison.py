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

# Known race values to look for
RACE_VALUES = {'White', 'Black', 'Asian', 'Hispanic', 'American Indian', 'Pacific Islander', 'Other', 'Two or More'}

# Known ethnicity values to look for
ETHNICITY_VALUES = {'Hispanic or Latino', 'Not Hispanic or Latino'}


def parse_pid_line(pid_line):
    """Extract demographic fields from PID segment by looking for known values."""
    fields = pid_line.strip().split('|')
    
    # Extract by known positions for gender and DOB (these are standard)
    gender = fields[7].strip() if len(fields) > 7 else ''
    dob = fields[6].strip() if len(fields) > 6 else ''
    
    # Search for race and ethnicity by matching known values
    race = 'Unknown'
    ethnicity = 'Unknown'
    
    for field in fields:
        field_clean = field.strip()
        # Check for race
        if field_clean in RACE_VALUES:
            race = field_clean
        # Check for ethnicity
        if field_clean in ETHNICITY_VALUES:
            ethnicity = field_clean
    
    return {
        'gender': gender or 'Unknown',
        'dob': dob,
        'race': race,
        'ethnicity': ethnicity
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
    
    # Calculate similarity scores
    gender_match = similarity_score(gender_pcts, DELAWARE_CONSENSUS['gender'])
    race_match = similarity_score(race_pcts, DELAWARE_CONSENSUS['race'])
    ethnicity_match = similarity_score(ethnicity_pcts, DELAWARE_CONSENSUS['ethnicity'])
    age_match = similarity_score(age_pcts, DELAWARE_CONSENSUS['age_group'])
    
    # Generate report
    output_file = output_dir / "adt_delaware_comparison.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ADT MESSAGE DEMOGRAPHICS vs DELAWARE CENSUS\n")
        f.write("=" * 70 + "\n\n")
        
        # Summary Section
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total ADT Messages Analyzed: {len(adt_files)}\n")
        f.write(f"Total Patients: {len(genders)}\n\n")
        
        f.write("Similarity to Delaware Census (±5% threshold):\n")
        f.write(f"  Gender      : {'MATCH ✓' if gender_match else 'NO MATCH ✗'}\n")
        f.write(f"  Race        : {'MATCH ✓' if race_match else 'NO MATCH ✗'}\n")
        f.write(f"  Ethnicity   : {'MATCH ✓' if ethnicity_match else 'NO MATCH ✗'}\n")
        f.write(f"  Age Group   : {'MATCH ✓' if age_match else 'NO MATCH ✗'}\n\n")
        
        matches = sum([gender_match, race_match, ethnicity_match, age_match])
        f.write(f"Overall: {matches}/4 demographic categories match Delaware consensus\n\n")
        
        f.write("=" * 70 + "\n\n")
        
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
        
        # Race Comparison
        f.write("RACE DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        for race, pct in sorted(race_pcts.items(), key=lambda x: -x[1]):
            f.write(f"  {race:20s}: {pct:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for race, pct in sorted(DELAWARE_CONSENSUS['race'].items(), key=lambda x: -x[1]):
            f.write(f"  {race:20s}: {pct:6.1f}%\n")
        similar = similarity_score(race_pcts, DELAWARE_CONSENSUS['race'])
        f.write(f"\nWithin ±5% threshold: {'YES ✓' if similar else 'NO ✗'}\n\n")
        
        # Ethnicity Comparison
        f.write("ETHNICITY DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        for eth, pct in sorted(ethnicity_pcts.items(), key=lambda x: -x[1]):
            f.write(f"  {eth:30s}: {pct:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for eth, pct in sorted(DELAWARE_CONSENSUS['ethnicity'].items(), key=lambda x: -x[1]):
            f.write(f"  {eth:30s}: {pct:6.1f}%\n")
        similar = similarity_score(ethnicity_pcts, DELAWARE_CONSENSUS['ethnicity'])
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
