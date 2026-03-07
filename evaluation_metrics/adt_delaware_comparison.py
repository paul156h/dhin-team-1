#!/usr/bin/env python3
"""
Compare ADT message demographics to Delaware census percentages.
Outputs race, ethnicity, gender, and age group distributions.
"""

import re
from datetime import date as _date
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

# HL7 Table 0001 — Administrative Sex
_VALID_GENDER = {'M', 'F', 'O', 'U', 'A', 'N', 'C'}

# HL7 Table 0005 / CDC OMB — Race codes and text equivalents
_VALID_RACE = {
    '1002-5', '2028-9', '2054-5', '2076-8', '2106-3', '2131-1', 'ASKU', 'UNK',
    'White', 'Black', 'Asian', 'Hispanic',
    'American Indian', 'Pacific Islander', 'Other', 'Two or More',
    'Black or African American', 'Two or more races',
    'Native Hawaiian or Other Pacific Islander',
    'American Indian or Alaska Native',
    'Unknown', 'Declined to state', 'Not Reported',
}

# HL7 Table 0189 / CDC PHIN — Ethnicity codes and text equivalents
_VALID_ETHNICITY = {
    '2135-2', '2186-5', 'ASKU', 'UNK',
    'Hispanic or Latino', 'Not Hispanic or Latino',
    'Unknown', 'Declined to state', 'Not Reported',
}

_DATE_RE = re.compile(r'^\d{8}')


def _validate_date(value: str, field_name: str):
    if not value:
        return True, ''
    raw = value.strip()
    if not _DATE_RE.match(raw):
        return False, f"{field_name}: '{raw}' is not in YYYYMMDD format"
    date_part = raw[:8]
    try:
        year  = int(date_part[0:4])
        month = int(date_part[4:6])
        day   = int(date_part[6:8])
        parsed = _date(year, month, day)
        today  = _date.today()
        if year < 1900:
            return False, f"{field_name}: year {year} is before 1900"
        if 'dob' in field_name.lower() or 'birth' in field_name.lower():
            if parsed > today:
                return False, f"{field_name}: {raw} is in the future"
        elif parsed.year > today.year + 1:
            return False, f"{field_name}: {raw} is unreasonably far in the future"
        return True, ''
    except ValueError as exc:
        return False, f"{field_name}: '{raw}' is not a valid calendar date — {exc}"


def _validate_gender(value: str):
    if not value:
        return True, ''
    code = value.strip()
    if code.upper() not in {c.upper() for c in _VALID_GENDER}:
        valid = ', '.join(sorted(_VALID_GENDER))
        return False, f"Gender '{code}' not in HL7 Table 0001 — valid: {valid}"
    return True, ''


def _validate_race(value: str):
    if not value:
        return True, ''
    code = value.strip()
    if code not in _VALID_RACE:
        return False, f"Race '{code}' not in HL7 Table 0005 / CDC OMB value set"
    return True, ''


def _validate_ethnicity(value: str):
    if not value:
        return True, ''
    code = value.strip()
    if code not in _VALID_ETHNICITY:
        return False, f"Ethnicity '{code}' not in HL7 Table 0189 / CDC PHIN value set"
    return True, ''


def parse_pid_line(pid_line):
    fields = pid_line.strip().split('|')

    gender = fields[7].strip() if len(fields) > 7 else ''
    dob = fields[6].strip() if len(fields) > 6 else ''

    race = 'Unknown'
    ethnicity = 'Unknown'

    for field in fields:
        field_clean = field.strip()
        if field_clean in RACE_VALUES:
            race = field_clean
        if field_clean in ETHNICITY_VALUES:
            ethnicity = field_clean

    validation_issues = []

    ok, msg = _validate_date(dob, 'PID DOB')
    if not ok:
        validation_issues.append(msg)

    ok, msg = _validate_gender(gender)
    if not ok:
        validation_issues.append(f"PID.8 (gender): {msg}")

    ok, msg = _validate_race(race)
    if not ok and race != 'Unknown':
        validation_issues.append(f"PID race: {msg}")

    ok, msg = _validate_ethnicity(ethnicity)
    if not ok and ethnicity != 'Unknown':
        validation_issues.append(f"PID ethnicity: {msg}")

    return {
        'gender': gender or 'Unknown',
        'dob': dob,
        'race': race,
        'ethnicity': ethnicity,
        'validation_issues': validation_issues,
    }


def get_age_group(dob):
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
    if not items:
        return {}
    
    counter = Counter(items)
    total = len(items)
    
    return {k: round(v / total * 100, 1) for k, v in counter.items()}


def similarity_score(adt_dist, delaware_dist, threshold=5.0):
    matches = 0
    total_categories = len(delaware_dist)
    
    for category, delaware_pct in delaware_dist.items():
        adt_pct = adt_dist.get(category, 0)
        if abs(adt_pct - delaware_pct) <= threshold:
            matches += 1
    
    return matches == total_categories


def main():
    project_root = Path(__file__).parent.parent
    adt_dir = project_root / "test_data" / "adt_messages"
    output_dir = project_root / "outputs"
    
    if not adt_dir.exists():
        print(f"Error: ADT directory not found: {adt_dir}")
        return
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    genders = []
    races = []
    ethnicities = []
    age_groups = []
    all_validation_issues = []

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
                    for issue in demographics['validation_issues']:
                        all_validation_issues.append((adt_file.name, issue))
                    break  # Only first PID per message
    
    gender_pcts = calculate_percentages(genders)
    race_pcts = calculate_percentages(races)
    ethnicity_pcts = calculate_percentages(ethnicities)
    age_pcts = calculate_percentages(age_groups)

    gender_match = similarity_score(gender_pcts, DELAWARE_CONSENSUS['gender'])
    race_match = similarity_score(race_pcts, DELAWARE_CONSENSUS['race'])
    ethnicity_match = similarity_score(ethnicity_pcts, DELAWARE_CONSENSUS['ethnicity'])
    age_match = similarity_score(age_pcts, DELAWARE_CONSENSUS['age_group'])

    output_file = output_dir / "adt_delaware_comparison.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ADT MESSAGE DEMOGRAPHICS vs DELAWARE CENSUS\n")
        f.write("=" * 70 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total ADT Messages Analyzed: {len(adt_files)}\n")
        f.write(f"Total Patients: {len(genders)}\n\n")
        
        f.write("Similarity to Delaware Census (+/-5% threshold):\n")
        f.write(f"  Gender      : {'MATCH' if gender_match else 'NO MATCH'}\n")
        f.write(f"  Race        : {'MATCH' if race_match else 'NO MATCH'}\n")
        f.write(f"  Ethnicity   : {'MATCH' if ethnicity_match else 'NO MATCH'}\n")
        f.write(f"  Age Group   : {'MATCH' if age_match else 'NO MATCH'}\n\n")
        
        matches = sum([gender_match, race_match, ethnicity_match, age_match])
        f.write(f"Overall: {matches}/4 demographic categories match Delaware consensus\n\n")
        
        f.write("=" * 70 + "\n\n")

        f.write("GENDER DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        for gender, pct in sorted(gender_pcts.items()):
            f.write(f"  {gender:20s}: {pct:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for gender, pct in sorted(DELAWARE_CONSENSUS['gender'].items()):
            f.write(f"  {gender:20s}: {pct:6.1f}%\n")
        similar = similarity_score(gender_pcts, DELAWARE_CONSENSUS['gender'])
        f.write(f"\nWithin +/-5% threshold: {'YES' if similar else 'NO'}\n\n")

        f.write("RACE DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        for race, pct in sorted(race_pcts.items(), key=lambda x: -x[1]):
            f.write(f"  {race:20s}: {pct:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for race, pct in sorted(DELAWARE_CONSENSUS['race'].items(), key=lambda x: -x[1]):
            f.write(f"  {race:20s}: {pct:6.1f}%\n")
        similar = similarity_score(race_pcts, DELAWARE_CONSENSUS['race'])
        f.write(f"\nWithin +/-5% threshold: {'YES' if similar else 'NO'}\n\n")

        f.write("ETHNICITY DISTRIBUTION\n")
        f.write("-" * 70 + "\n")
        f.write("ADT Percentages:\n")
        for eth, pct in sorted(ethnicity_pcts.items(), key=lambda x: -x[1]):
            f.write(f"  {eth:30s}: {pct:6.1f}%\n")
        f.write("\nDelaware Consensus:\n")
        for eth, pct in sorted(DELAWARE_CONSENSUS['ethnicity'].items(), key=lambda x: -x[1]):
            f.write(f"  {eth:30s}: {pct:6.1f}%\n")
        similar = similarity_score(ethnicity_pcts, DELAWARE_CONSENSUS['ethnicity'])
        f.write(f"\nWithin +/-5% threshold: {'YES' if similar else 'NO'}\n\n")

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
        f.write(f"\nWithin +/-5% threshold: {'YES' if similar else 'NO'}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Note: A threshold of +/-5% is used for similarity comparison.\n")
        f.write("=" * 70 + "\n\n")

        # Content Validation Summary
        f.write("CONTENT VALIDATION SUMMARY\n")
        f.write("-" * 70 + "\n")
        total_checks = len(genders) * 4  # date, gender, race, ethnicity per patient
        if all_validation_issues:
            f.write(f"Issues found: {len(all_validation_issues)} across {len(adt_files)} messages\n\n")
            for filename, issue in sorted(all_validation_issues):
                f.write(f"  [{filename}] {issue}\n")
        else:
            f.write("No content validation issues detected.\n")
        f.write("\nValidation checks performed per patient record:\n")
        f.write("  - DOB: YYYYMMDD format, year >= 1900, not in future\n")
        f.write("  - Gender: HL7 Table 0001 codes (M, F, O, U, A, N, C)\n")
        f.write("  - Race: HL7 Table 0005 / CDC OMB codes and text values\n")
        f.write("  - Ethnicity: HL7 Table 0189 / CDC PHIN codes and text values\n")
        f.write("=" * 70 + "\n")

    print(f"Comparison report saved to: {output_file}")


if __name__ == "__main__":
    main()
