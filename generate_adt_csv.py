#!/usr/bin/env python3
"""
Generate a synthetic CSV that matches a given schema (columns + order) and fills
non-time fields using lookup tables. Time-related and name-like fields are left blank.
"""

import argparse
import random
import re
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import pandas as pd



SEX = ["M", "F"]
SEX_WEIGHTS = [0.487, 0.513]  # Delaware sex distribution

RACE_DESC = [
    "White",
    "Black or African American",
    "Asian",
    "Two or More Races",
    "Other",
]
RACE_WEIGHTS = [0.600, 0.224, 0.040, 0.029, 0.107]  # Delaware non-Hispanic race distribution

ETHNICITY = ["Not Hispanic or Latino", "Hispanic or Latino"]
ETHNICITY_WEIGHTS = [0.909, 0.091]  # Delaware ethnicity distribution

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

# Small synthetic lookups for names, zip codes, providers, services
FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
DE_ZIPS = [19801, 19702, 19901, 19904, 19703, 19805, 19977]
PROVIDERS = [
    ("1000000010", "Adams", "Eleanor"),
    ("1000000020", "Baker", "Thomas"),
    ("1000000030", "Clark", "Susan"),
]
SERVICES = ["ED", "MED", "SURG", "OBS", "PEDS"]
ASSIGNING_AUTHORITY = "DEMED"

# Basic HL7 code mappings (approximate/placeholder CE values)
RACE_CODE_MAP = {
    "White": "2106-3^White^HL70005",
    "Black or African American": "2054-5^Black or African American^HL70005",
    "Asian": "2028-9^Asian^HL70005",
    "Two or More Races": "2131-1^Two or More Races^HL70005",
    "Other": "9999-9^Other^HL70005",
}
ETHNICITY_CODE_MAP = {
    "Not Hispanic or Latino": "2186-5^Not Hispanic or Latino^HL70189",
    "Hispanic or Latino": "2135-2^Hispanic or Latino^HL70189",
}
LANG_MAP = {"ENGL": "eng^English", "SPAN": "spa^Spanish"}

def contains(col: str, *subs: str) -> bool:
    c = col.lower()
    return any(s in c for s in subs)

def should_blank_time(col: str) -> bool:
    # Previously this tool left time fields blank on purpose; for ADT generation we
    # now populate time fields. Keep the function for compatibility but always
    # return False so times are filled.
    return False

def should_blank_name(col: str) -> bool:
    # Allow name-like fields to be populated for HL7 PID segments.
    return False


def generate_name(seed=None):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return first, last

def generate_dob_from_age(age_years: int):
    # Derive a DOB from age with small jitter in days
    today = datetime.utcnow().date()
    years = age_years
    # jitter +/- 2 years in days
    jitter_days = random.randint(-365 * 2, 365 * 2)
    try:
        dob = datetime(today.year - years, random.randint(1, 12), random.randint(1, 28)).date()
    except Exception:
        dob = today.replace(year=max(1900, today.year - years))
    dob = dob + timedelta(days=jitter_days)
    return dob

def rand_datetime_between(start: datetime, end: datetime):
    span = end - start
    sec = random.randint(0, int(span.total_seconds()))
    return start + timedelta(seconds=sec)

def message_control_id():
    return uuid.uuid4().hex

def gen_phone():
    return f"302-555-{random.randint(1000,9999):04d}"

def gen_address():
    street_no = random.randint(100, 9999)
    street = f"{street_no} Main St"
    city = "Wilmington"
    state = "DE"
    zipc = random.choice(DE_ZIPS)
    return f"{street}^{city}^{state}^{zipc}"

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
    
    # Delaware age distribution (2020 Census)
    # Weighted by actual age ranges from Delaware demographics
    age_bins = [
        (0, 17, 0.20),      # Under 18: 20%
        (18, 34, 0.22),     # 18-34: 22%
        (35, 49, 0.21),     # 35-49: 21%
        (50, 64, 0.20),     # 50-64: 20%
        (65, 79, 0.12),     # 65-79: 12%
        (80, 100, 0.05)     
    ]
    # Choose age bin based on Delaware distribution
    age_bin_choice = weighted_choice(age_bins, [b[2] for b in age_bins])
    age_years = random.randint(age_bin_choice[0], age_bin_choice[1])

    # Use Delaware census weights
    sex = weighted_choice(SEX, SEX_WEIGHTS)
    race = weighted_choice(RACE_DESC, RACE_WEIGHTS)
    ethnicity = weighted_choice(ETHNICITY, ETHNICITY_WEIGHTS)

    dx_code, dx_desc = random.choice(ICD10_COMMON)
    admission_type = random.choice(ADMISSION_TYPES)
    admission_loc = random.choice(ADMISSION_LOCS)
    discharge_loc = random.choice(DISCHARGE_LOCS)
    insurance = random.choice(INSURANCE)
    language = random.choice(LANG)
    religion = random.choice(RELIGION)
    marital = random.choice(MARITAL)


    # Compose synthetic name/dob/provider/timestamps for mapping into HL7-like columns
    first_name, last_name = generate_name()
    dob = generate_dob_from_age(age_years)
    now = datetime.utcnow()
    msg_dt = now
    evn_dt = now

    # Admission/admit times: choose admit sometime in the recent past
    admit_offset_days = random.randint(0, 30)
    admittime = now - timedelta(days=admit_offset_days, hours=random.randint(0, 72))
    # Discharge time: for inpatient assume short stay, otherwise blank for register
    if admission_type in ("EMERGENCY", "TRAUMA CENTER", "OBSERVATION"):
        # ED/observation — short stay
        dischtime = admittime + timedelta(hours=random.randint(1, 48))
    else:
        # elective/urgent/newborn — may have longer stay
        dischtime = admittime + timedelta(days=random.randint(0, 7), hours=random.randint(0, 23))

    # pick a provider
    prov_id, prov_last, prov_first = random.choice(PROVIDERS)

    row = {}
    for col in columns:
        c = col.lower()
        value = ""

        # IDs and HL7-styled identifier fields
        if contains(c, "msh-3") or contains(c, "sendingapplication") or contains(c, "sending_app"):
            value = "SYNTH_APP"
        elif contains(c, "msh-4") or contains(c, "sendingfacility"):
            value = "SYNTH_FAC"
        elif contains(c, "msh-5") or contains(c, "receivingapplication"):
            value = "REC_APP"
        elif contains(c, "msh-6") or contains(c, "receivingfacility"):
            value = "REC_FAC"
        elif contains(c, "msh-7") or contains(c, "message_datetime") or contains(c, "message_date"):
            value = msg_dt.strftime("%Y%m%d%H%M%S")
        elif contains(c, "msh-9") or contains(c, "messagetype"):
            value = "ADT^A04"
        elif contains(c, "msh-10") or contains(c, "messagecontrol"):
            value = message_control_id()
        elif contains(c, "msh-11") or contains(c, "processingid"):
            value = "P"
        elif contains(c, "msh-12") or contains(c, "versionid"):
            value = "2.5.1"

        # EVN
        elif contains(c, "evn-1") or contains(c, "eventtype"):
            value = "A04"
        elif contains(c, "evn-2") or contains(c, "eventdatetime"):
            value = evn_dt.strftime("%Y%m%d%H%M%S")

        # IDs
        elif contains(c, "subject_id", "subjectid", "patientid", "mrn", "pid3"):
            # Format as a full PID-3 value: id^^^AssigningAuthority^MR
            value = f"{subj_id}^^^{ASSIGNING_AUTHORITY}^MR"
        elif contains(c, "hadm_id", "hadmid", "visitid", "encounter", "pv1"):
            value = hadm_id

        # Patient demographics
        elif contains(c, "gender", "sex", "administrativesex"):
            value = sex
        elif contains(c, "race"):
            value = RACE_CODE_MAP.get(race, race)
        elif contains(c, "ethnicity", "ethnic", "hispanic"):
            value = ETHNICITY_CODE_MAP.get(ethnicity, ethnicity)
        elif contains(c, "age"):
            value = age_years
        elif contains(c, "dob") or contains(c, "birth") or contains(c, "pid-7"):
            value = dob.strftime("%Y%m%d")

        # Name fields
        elif contains(c, "patientname") or contains(c, "pid-5") or contains(c, "name") or contains(c, "given"):
            # HL7 XPN-like: Family^Given
            value = f"{last_name}^{first_name}"
        elif contains(c, "family") or contains(c, "lastname") or contains(c, "surname"):
            value = last_name
        elif contains(c, "first") or contains(c, "givenname"):
            value = first_name

        # Address / phone
        elif contains(c, "address") or contains(c, "pid-11"):
            value = gen_address()
        elif contains(c, "phone") or contains(c, "pid-13"):
            value = gen_phone()

        # Admission / Discharge metadata
        elif contains(c, "admissiontype", "patientclass") or contains(c, "pv1-2"):
            # For register (A04) patient class is Outpatient
            value = "O"
        elif contains(c, "admissionlocation", "admissionsource", "admitloc") or contains(c, "pv1-14"):
            value = admission_loc
        elif contains(c, "dischargelocation", "dischloc") or contains(c, "pv1-36"):
            value = discharge_loc
        elif contains(c, "pv1-3") or contains(c, "assignedlocation"):
            # Facility^Building^Room-Bed
            value = f"SYNTH_FAC^WARD1^{random.randint(100,499)}-{random.randint(1,4)}"

        # Providers
        elif contains(c, "attending") or contains(c, "pv1-7"):
            value = f"{prov_id}^{prov_last}^{prov_first}^Dr^NPI"
        elif contains(c, "referring") or contains(c, "pv1-8"):
            # pick another provider
            pid2, plast2, pfirst2 = random.choice(PROVIDERS)
            value = f"{pid2}^{plast2}^{pfirst2}^Dr^NPI"
        elif contains(c, "pv1-10") or contains(c, "hospitalservice"):
            value = random.choice(SERVICES)

        # PV1 admit/discharge datetimes
        elif contains(c, "pv1-44") or contains(c, "admittime"):
            value = admittime.strftime("%Y%m%d%H%M%S")
        elif contains(c, "pv1-45") or contains(c, "dischtime") or contains(c, "dischargedatetime"):
            value = dischtime.strftime("%Y%m%d%H%M%S")

        # Diagnosis
        elif contains(c, "diagnosiscode", "dg1", "dx_code", "dxcode"):
            # DG1-3 CE style
            value = f"{dx_code}^{dx_desc}^ICD-10"
        elif contains(c, "diagnosis", "dxdesc"):
            value = dx_desc

        # Insurance / IN1
        elif contains(c, "insurance", "payer") or contains(c, "in1"):
            # Build a small IN1-style string: PlanID^PlanName^PolicyNumber
            policy = rand_digits(9)
            value = f"{insurance}Plan^{insurance}^{policy}"
        elif contains(c, "language", "lang"):
            value = LANG_MAP.get(language, language)
        elif contains(c, "religion", "rel"):
            value = religion
        elif contains(c, "marital"):
            value = marital

        # NK1 (next-of-kin) basic fields
        elif contains(c, "nk1") or contains(c, "nextofkin"):
            # Simple NK1 name/relationship/phone
            nk_first, nk_last = generate_name()
            value = f"{nk_last}^{nk_first}^SPO^{gen_phone()}"

        # HL7-ish meta (non-time) defaults
        elif contains(c, "messagetype", "msh9"):
            value = "ADT^A04"
        elif contains(c, "processingid", "msh11"):
            value = "P"
        elif contains(c, "versionid", "msh12"):
            value = "2.5.1"

        # Fallbacks: existing simple mappings from previous script
        elif contains(c, "gender", "sex"):
            value = sex
        elif contains(c, "race"):
            value = race
        elif contains(c, "ethnicity"):
            value = ethnicity

        # Admissions counts
        elif contains(c, "recent admissions", "admissions_count", "num_admissions", "admissions"):
            value = random.randint(0, 10)

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