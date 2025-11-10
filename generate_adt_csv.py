#!/usr/bin/env python3
"""
Generate a synthetic CSV that matches a given schema (columns + order) and fills
non-time fields using lookup tables. Time-related and name-like fields are left blank.
"""

import argparse
import random
import re
import string
from pathlib import Path
from typing import List
import pandas as pd

# Lookup tables (variety; no cities/addresses)
SEX = ["M", "F", "O"]
RACE_DESC = [
    "White",
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander",
    "Some Other Race",
    "Two or More Races",
]
ETHNICITY = ["Not Hispanic or Latino", "Hispanic or Latino"]

ADMISSION_TYPES = ["EMERGENCY", "ELECTIVE", "URGENT", "NEWBORN", "TRAUMA CENTER", "OBSERVATION"]
ADMISSION_LOCS = [
    "EMERGENCY ROOM", "PHYSICIAN REFERRAL", "CLINIC REFERRAL",
    "TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING",
    "HMO REFERRAL", "WALK-IN", "AMBULANCE"
]
DISCHARGE_LOCS = [
    "HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY",
    "REHAB", "LONG TERM CARE HOSPITAL", "AGAINST MEDICAL ADVICE",
    "TRANSFER TO ANOTHER HOSPITAL"
]
INSURANCE = ["Medicare", "Medicaid", "Private", "Self Pay", "Government", "No Charge", "Other"]
LANG = ["ENGL", "SPAN", "MAND", "CANT", "VIET", "ARAB", "FREN", "RUSS", "KORE", "HIND", "PORT", "GERM"]
RELIGION = ["CATHOLIC", "PROTESTANT", "JEWISH", "BUDDHIST", "HINDU", "MUSLIM", "CHRISTIAN", "ORTHODOX", "ATHEIST", "AGNOSTIC", "OTHER"]
MARITAL = ["MARRIED", "SINGLE", "DIVORCED", "WIDOWED", "SEPARATED", "PARTNERED", "UNKNOWN"]

ICD10_COMMON = [
    ("R10.9", "Unspecified abdominal pain"), ("J06.9", "Acute upper respiratory infection, unspecified"),
    ("M54.5", "Low back pain"), ("I10", "Essential (primary) hypertension"),
    ("E11.9", "Type 2 diabetes mellitus without complications"), ("R07.9", "Chest pain, unspecified"),
    ("N39.0", "Urinary tract infection, site not specified"), ("K52.9", "Noninfective gastroenteritis and colitis, unspecified"),
    ("J45.909", "Unspecified asthma, uncomplicated"), ("S09.90XA", "Unspecified injury of head, initial encounter"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis"), ("R51.9", "Headache, unspecified"),
    ("H10.9", "Unspecified conjunctivitis"), ("L03.90", "Cellulitis, unspecified"),
    ("A09", "Infectious gastroenteritis and colitis, unspecified"), ("R50.9", "Fever, unspecified"),
    ("R55", "Syncope and collapse"), ("R42", "Dizziness and giddiness"),
    ("J20.9", "Acute bronchitis, unspecified"), ("S16.1XXA", "Strain of muscle, fascia and tendon at neck level"),
]

TIME_TOKENS = [
    "time", "date", "datetime", "admit", "disch", "edreg", "edout",
    "intime", "outtime", "charttime", "storetime", "starttime", "endtime",
    "chartdate", "birth", "dob"
]
NAME_TOKENS = ["name", "first", "last", "given", "family", "middle"]

def contains(col: str, *subs: str) -> bool:
    c = col.lower()
    return any(s in c for s in subs)

def should_blank_time(col: str) -> bool:
    c = col.lower()
    return any(tok in c for tok in TIME_TOKENS)

def should_blank_name(col: str) -> bool:
    c = col.lower()
    return any(tok in c for tok in NAME_TOKENS)

def rand_digits(n: int) -> str:
    import string
    return "".join(random.choices(string.digits, k=n))

def weighted_choice(items: List, weights: List[float] = None):
    if weights is None:
        return random.choice(items)
    return random.choices(items, weights=weights, k=1)[0]

def generate_row(columns: List[str]) -> dict:
    subj_id = int(rand_digits(7))
    hadm_id = int(rand_digits(7))
    age_years = random.randint(0, 95)  

    sex = weighted_choice(SEX, [0.49, 0.49, 0.02])
    race = weighted_choice(RACE_DESC, [0.61, 0.12, 0.06, 0.01, 0.01, 0.12, 0.07])
    ethnicity = weighted_choice(ETHNICITY, [0.83, 0.17])

    dx_code, dx_desc = random.choice(ICD10_COMMON)
    admission_type = random.choice(ADMISSION_TYPES)
    admission_loc = random.choice(ADMISSION_LOCS)
    discharge_loc = random.choice(DISCHARGE_LOCS)
    insurance = random.choice(INSURANCE)
    language = random.choice(LANG)
    religion = random.choice(RELIGION)
    marital = random.choice(MARITAL)

    row = {}
    for col in columns:
        c = col.lower()
        value = ""

        # Always blank time-related and name-like fields
        if should_blank_time(c) or should_blank_name(c):
            row[col] = ""
            continue

        # IDs
        if contains(c, "subject_id", "subjectid", "patientid", "mrn", "pid3"):
            value = subj_id
        elif contains(c, "hadm_id", "hadmid", "visitid", "encounter", "pv1"):
            value = hadm_id

        # Demographics
        elif contains(c, "gender", "sex", "administrativesex"):
            value = sex
        elif contains(c, "race"):
            value = race
        elif contains(c, "ethnicity", "ethnic", "hispanic"):  
            value = ethnicity
        elif contains(c, "age"):
            value = age_years

        # Admission / Discharge metadata (non-time)
        elif contains(c, "admissiontype", "patientclass"):
            value = admission_type
        elif contains(c, "admissionlocation", "admissionsource", "admitloc"):
            value = admission_loc
        elif contains(c, "dischargelocation", "dischloc"):
            value = discharge_loc

        # Admissions counts
        elif contains(c, "recent admissions", "admissions_count", "num_admissions", "admissions"):
            value = random.randint(0, 10)

        # Insurance / misc
        elif contains(c, "insurance", "payer"):
            value = insurance
        elif contains(c, "language", "lang"):
            value = language
        elif contains(c, "religion", "rel"):
            value = religion
        elif contains(c, "marital"):
            value = marital

        # Diagnosis
        elif contains(c, "diagnosiscode", "dg1", "dx_code", "dxcode"):
            value = dx_code
        elif contains(c, "diagnosis", "dxdesc"):
            value = dx_desc

        # HL7-ish meta (non-time)
        elif contains(c, "messagetype", "msh9"):
            value = "ADT^A04"
        elif contains(c, "processingid", "msh11"):
            value = "P"
        elif contains(c, "versionid", "msh12"):
            value = "2.5.1"

        # Otherwise leave blank
        row[col] = value

    return row

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic admissions CSV matching a schema (times blank; ethnicity lookup).")
    parser.add_argument("--schema", type=Path, required=True, help="Path to schema CSV (to copy column names/order).")
    parser.add_argument("--out", type=Path, default=Path("generated_csvs/generated_data.csv"), help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    df_schema_sample = pd.read_csv(args.schema, nrows=5)
    columns = list(df_schema_sample.columns)

    rows = [generate_row(columns) for _ in range(args.rows)]
    out_df = pd.DataFrame(rows, columns=columns)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    filled = sum(out_df[c].astype(str).ne("").any() for c in columns)
    print(f"✅ Generated {args.rows} rows → {args.out}")
    print(f"Columns with data: {filled}/{len(columns)}")
    still_blank = [c for c in columns if out_df[c].astype(str).eq("").all()]
    if still_blank:
        print("Columns left blank (by rule or no mapping found):")
        for c in still_blank:
            print(f"  - {c}")

if __name__ == "__main__":
    main()
