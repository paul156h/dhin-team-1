#!/usr/bin/env python3
"""
All-in-one synthetic ADT generator: produces both a CSV and HL7 v2.5.1 ADT messages
from a single command. This combines generate_adt_csv.py and generate_adt_hl7.py.

Usage:
  python generate_synthetic_adt.py --schema schema_hl7_minimal.csv --rows 100 --seed 42 --csv-out generated.csv --hl7-out messages.hl7
"""

import argparse
import random
import uuid
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import hl7

#input parameters : patient count, age ranges, gender, race and ethnicity


SEX = ["M", "F"]
SEX_WEIGHTS = [0.487, 0.513]

RACE_DESC = [
    "White",
    "Black or African American",
    "Asian",
    "Two or More Races",
    "Other",
]
RACE_WEIGHTS = [0.600, 0.224, 0.040, 0.029, 0.107]

ETHNICITY = ["Not Hispanic or Latino", "Hispanic or Latino"]
ETHNICITY_WEIGHTS = [0.909, 0.091]

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


def generate_name():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return first, last


def generate_dob_from_age(age_years: int):
    today = datetime.utcnow().date()
    years = age_years
    jitter_days = random.randint(-365 * 2, 365 * 2)
    try:
        dob = datetime(today.year - years, random.randint(1, 12), random.randint(1, 28)).date()
    except Exception:
        dob = today.replace(year=max(1900, today.year - years))
    dob = dob + timedelta(days=jitter_days)
    return dob


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
    return "".join(random.choices("0123456789", k=n))


def weighted_choice(items: List, weights: List[float] = None):
    if weights is None:
        return random.choice(items)
    return random.choices(items, weights=weights, k=1)[0]


def generate_row(columns: List[str], age_min: int = None, age_max: int = None,
                   gender_list: List[str] = None, gender_weights: List[float] = None,
                   race_list: List[str] = None, race_weights: List[float] = None,
                   ethnicity_list: List[str] = None, ethnicity_weights: List[float] = None) -> dict:
    subj_id = int(rand_digits(7))
    hadm_id = int(rand_digits(7))
    
    # Use default age bins or custom range
    if age_min is not None and age_max is not None:
        age_years = random.randint(age_min, age_max)
    else:
        age_bins = [
            (0, 17, 0.20),
            (18, 34, 0.22),
            (35, 49, 0.21),
            (50, 64, 0.20),
            (65, 79, 0.12),
            (80, 100, 0.05)
        ]
        age_bin_choice = weighted_choice(age_bins, [b[2] for b in age_bins])
        age_years = random.randint(age_bin_choice[0], age_bin_choice[1])

    # Use provided gender/race/ethnicity lists or defaults
    sex = weighted_choice(gender_list or SEX, gender_weights or SEX_WEIGHTS)
    race = weighted_choice(race_list or RACE_DESC, race_weights or RACE_WEIGHTS)
    ethnicity = weighted_choice(ethnicity_list or ETHNICITY, ethnicity_weights or ETHNICITY_WEIGHTS)

    dx_code, dx_desc = random.choice(ICD10_COMMON)
    admission_type = random.choice(ADMISSION_TYPES)
    admission_loc = random.choice(ADMISSION_LOCS)
    discharge_loc = random.choice(DISCHARGE_LOCS)
    insurance = random.choice(INSURANCE)
    language = random.choice(LANG)
    religion = random.choice(RELIGION)
    marital = random.choice(MARITAL)

    first_name, last_name = generate_name()
    dob = generate_dob_from_age(age_years)
    now = datetime.utcnow()
    msg_dt = now
    evn_dt = now

    admit_offset_days = random.randint(0, 30)
    admittime = now - timedelta(days=admit_offset_days, hours=random.randint(0, 72))
    if admission_type in ("EMERGENCY", "TRAUMA CENTER", "OBSERVATION"):
        dischtime = admittime + timedelta(hours=random.randint(1, 48))
    else:
        dischtime = admittime + timedelta(days=random.randint(0, 7), hours=random.randint(0, 23))

    prov_id, prov_last, prov_first = random.choice(PROVIDERS)

    row = {}
    for col in columns:
        c = col.lower()
        value = ""

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

        elif contains(c, "evn-1") or contains(c, "eventtype"):
            value = "A04"
        elif contains(c, "evn-2") or contains(c, "eventdatetime"):
            value = evn_dt.strftime("%Y%m%d%H%M%S")

        elif contains(c, "subject_id", "subjectid", "patientid", "mrn", "pid3"):
            value = f"{subj_id}^^^{ASSIGNING_AUTHORITY}^MR"
        elif contains(c, "hadm_id", "hadmid", "visitid", "encounter", "pv1"):
            value = hadm_id

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

        elif contains(c, "patientname") or contains(c, "pid-5") or contains(c, "name") or contains(c, "given"):
            value = f"{last_name}^{first_name}"
        elif contains(c, "family") or contains(c, "lastname") or contains(c, "surname"):
            value = last_name
        elif contains(c, "first") or contains(c, "givenname"):
            value = first_name

        elif contains(c, "address") or contains(c, "pid-11"):
            value = gen_address()
        elif contains(c, "phone") or contains(c, "pid-13"):
            value = gen_phone()

        elif contains(c, "admissiontype", "patientclass") or contains(c, "pv1-2"):
            value = "O"
        elif contains(c, "admissionlocation", "admissionsource", "admitloc") or contains(c, "pv1-14"):
            value = admission_loc
        elif contains(c, "dischargelocation", "dischloc") or contains(c, "pv1-36"):
            value = discharge_loc
        elif contains(c, "pv1-3") or contains(c, "assignedlocation"):
            value = f"SYNTH_FAC^WARD1^{random.randint(100,499)}-{random.randint(1,4)}"

        elif contains(c, "attending") or contains(c, "pv1-7"):
            value = f"{prov_id}^{prov_last}^{prov_first}^Dr^NPI"
        elif contains(c, "referring") or contains(c, "pv1-8"):
            pid2, plast2, pfirst2 = random.choice(PROVIDERS)
            value = f"{pid2}^{plast2}^{pfirst2}^Dr^NPI"
        elif contains(c, "pv1-10") or contains(c, "hospitalservice"):
            value = random.choice(SERVICES)

        elif contains(c, "pv1-44") or contains(c, "admittime"):
            value = admittime.strftime("%Y%m%d%H%M%S")
        elif contains(c, "pv1-45") or contains(c, "dischtime") or contains(c, "dischargedatetime"):
            value = dischtime.strftime("%Y%m%d%H%M%S")

        elif contains(c, "diagnosiscode", "dg1", "dx_code", "dxcode"):
            value = f"{dx_code}^{dx_desc}^ICD-10"
        elif contains(c, "diagnosis", "dxdesc"):
            value = dx_desc

        elif contains(c, "insurance", "payer") or contains(c, "in1"):
            policy = rand_digits(9)
            value = f"{insurance}Plan^{insurance}^{policy}"
        elif contains(c, "language", "lang"):
            value = LANG_MAP.get(language, language)
        elif contains(c, "religion", "rel"):
            value = religion
        elif contains(c, "marital"):
            value = marital

        elif contains(c, "nk1") or contains(c, "nextofkin"):
            nk_first, nk_last = generate_name()
            value = f"{nk_last}^{nk_first}^SPO^{gen_phone()}"

        elif contains(c, "messagetype", "msh9"):
            value = "ADT^A04"
        elif contains(c, "processingid", "msh11"):
            value = "P"
        elif contains(c, "versionid", "msh12"):
            value = "2.5.1"

        elif contains(c, "recent admissions", "admissions_count", "num_admissions", "admissions"):
            value = random.randint(0, 10)

        row[col] = value

    return row


def generate_csv(schema_path: Path, num_rows: int, age_min: int = None, age_max: int = None,
                 gender_list: List[str] = None, gender_weights: List[float] = None,
                 race_list: List[str] = None, race_weights: List[float] = None,
                 ethnicity_list: List[str] = None, ethnicity_weights: List[float] = None) -> pd.DataFrame:
    """Generate synthetic patient CSV data with optional demographic overrides."""
    df_schema = pd.read_csv(schema_path, nrows=1)
    columns = list(df_schema.columns)
    rows = [generate_row(columns, age_min, age_max, gender_list, gender_weights,
                         race_list, race_weights, ethnicity_list, ethnicity_weights)
            for _ in range(num_rows)]
    return pd.DataFrame(rows, columns=columns)


#  HL7 Conversion (from generate_adt_hl7.py

def fmt_ts(val):
    if pd.isna(val) or val == "":
        return ""
    s = str(val)
    if s.isdigit() and len(s) >= 8:
        return s[:14].ljust(14, '0')
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y%m%d%H%M%S")
        except Exception:
            continue
    return s


def make_msh(row):
    field_sep = '|'
    enc = '^~\\&'
    sending_app = row.get('msh-3') or row.get('sendingapplication') or 'SYNTH_APP'
    sending_fac = row.get('msh-4') or row.get('sendingfacility') or 'SYNTH_FAC'
    recv_app = row.get('msh-5') or row.get('receivingapplication') or 'REC_APP'
    recv_fac = row.get('msh-6') or row.get('receivingfacility') or 'REC_FAC'
    ts = fmt_ts(row.get('msh-7')) or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    msg_type = row.get('msh-9') or 'ADT^A04'
    ctrl = row.get('msh-10') or uuid.uuid4().hex
    proc = row.get('msh-11') or 'P'
    ver = row.get('msh-12') or '2.5.1'
    parts = ["MSH", field_sep + enc, sending_app, sending_fac, recv_app, recv_fac, ts, '', msg_type, ctrl, proc, ver]
    return field_sep.join(parts)


def make_evn(row):
    evn1 = row.get('evn-1') or row.get('eventtype') or 'A04'
    evn2 = fmt_ts(row.get('evn-2')) or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"EVN|{evn1}|{evn2}"


def make_pid(row):
    pid3 = row.get('subject_id') or row.get('pid-3') or ''
    name = row.get('pid-5') or row.get('patientname') or ''
    dob = fmt_ts(row.get('pid-7') or row.get('dob') or row.get('birth') or '')
    sex = row.get('gender') or row.get('sex') or ''
    race = row.get('race') or ''
    addr = row.get('pid-11') or row.get('address') or ''
    phone = row.get('pid-13') or row.get('phone') or ''
    ethnic = row.get('ethnicity') or row.get('pid-22') or ''
    parts = ["PID", "1", "", pid3, "", name, dob, sex, race, "", addr, phone, "", "", "", "", "", "", ethnic]
    return '|'.join(parts)


def make_pv1(row):
    pclass = row.get('pv1-2') or row.get('patientclass') or row.get('admissiontype') or 'O'
    assigned = row.get('pv1-3') or row.get('assignedlocation') or ''
    attending = row.get('pv1-7') or row.get('attending') or ''
    service = row.get('pv1-10') or row.get('hospitalservice') or ''
    visitnum = str(row.get('pv1-19') or row.get('hadm_id') or '')
    admit = fmt_ts(row.get('pv1-44') or row.get('admittime') or '')
    disc = fmt_ts(row.get('pv1-45') or row.get('dischtime') or row.get('dischargedatetime') or '')
    parts = ["PV1", "1", str(pclass), str(assigned), "", "", str(attending), "", "", str(service), "", "", "", "", "", "", "", "", "", visitnum, "", "", "", "", "", "", "", "", "", "", "", "", "", admit, disc]
    return '|'.join(parts)


def make_dg1(row):
    code = row.get('diagnosiscode') or row.get('dg1') or ''
    desc = row.get('diagnosis') or row.get('dxdesc') or ''
    if not code:
        return ''
    if '^' in code:
        ce = code
    else:
        ce = f"{code}^{desc}^ICD-10"
    return f"DG1|1||{ce}||"


def make_in1(row):
    in1 = row.get('in1') or row.get('insurance') or ''
    if not in1:
        return ''
    return f"IN1|1|{in1}"


def make_nk1(row):
    nk = row.get('nk1') or row.get('nextofkin') or ''
    if not nk:
        return ''
    return f"NK1|1|{nk}"


import hl7

def row_to_message(row):
    """Build an HL7 message using hl7py library."""
    r = {k: (v if pd.notna(v) else '') for k, v in row.items()}
    
    # Build segment strings (pipe-delimited fields, caret-delimited components)
    msh = make_msh(r)
    evn = make_evn(r)
    pid = make_pid(r)
    pv1 = make_pv1(r)
    dg1 = make_dg1(r)
    in1 = make_in1(r)
    nk1 = make_nk1(r)
    
    # Assemble segments into a single message
    segments = [msh, evn, pid]
    if pv1:
        segments.append(pv1)
    if dg1:
        segments.append(dg1)
    if in1:
        segments.append(in1)
    if nk1:
        segments.append(nk1)
    
    # Parse and re-serialize with hl7py for proper formatting
    msg_text = '\r'.join(segments)
    try:
        parsed = hl7.parse(msg_text)
        formatted = str(parsed)  # Re-serialize to ensure proper HL7 format
        return formatted + '\r'
    except Exception as e:
        # Fallback to plain message if parsing fails
        print(f"Warning: HL7 parsing failed for row, using plain format: {e}")
        return msg_text + '\r'

def df_to_hl7_messages(df: pd.DataFrame) -> List[str]:
    """Convert DataFrame rows to HL7 messages."""
    messages = []
    for _, row in df.iterrows():
        messages.append(row_to_message(row))
    return messages



#  Combined entry point


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Delaware ADT data: both CSV and HL7 v2.5.1 messages in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_synthetic_adt.py --schema schema_hl7_minimal.csv --rows 100
  python generate_synthetic_adt.py --schema schema_hl7_minimal.csv --rows 100 --seed 42 --csv-out data.csv --hl7-out messages.hl7
        """
    )
    parser.add_argument("--schema", type=Path, required=True, help="Schema CSV (columns/order template).")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    parser.add_argument("--csv-out", type=Path, default=Path("generated_adt.csv"), help="Output CSV file path.") 
    parser.add_argument("--hl7-out", type=Path, default=Path("generated_adt_messages.hl7"), help="Output HL7 file path.")
    
    # Demographic parameters
    parser.add_argument("--age-min", type=int, default=None, help="Minimum age (overrides default age distribution).")
    parser.add_argument("--age-max", type=int, default=None, help="Maximum age (overrides default age distribution).")
    parser.add_argument("--gender", type=str, nargs="+", default=None, help="Gender values to use (e.g., M F). Overrides default.")
    parser.add_argument("--gender-weights", type=float, nargs="+", default=None, help="Gender weights (e.g., 0.5 0.5). Must match --gender length.")
    parser.add_argument("--race", type=str, nargs="+", default=None, help="Race values to use (e.g., 'White' 'Black or African American'). Overrides default.")
    parser.add_argument("--race-weights", type=float, nargs="+", default=None, help="Race weights. Must match --race length.")
    parser.add_argument("--ethnicity", type=str, nargs="+", default=None, help="Ethnicity values (e.g., 'Hispanic or Latino' 'Not Hispanic or Latino'). Overrides default.")
    parser.add_argument("--ethnicity-weights", type=float, nargs="+", default=None, help="Ethnicity weights. Must match --ethnicity length.")
    
    args = parser.parse_args()

    random.seed(args.seed)
    
    # Validate and normalize demographic parameters
    if args.age_min is not None and args.age_max is not None:
        if args.age_min > args.age_max:
            parser.error("--age-min must be <= --age-max")
    
    if args.gender_weights and args.gender:
        if len(args.gender_weights) != len(args.gender):
            parser.error("--gender-weights length must match --gender length")
    
    if args.race_weights and args.race:
        if len(args.race_weights) != len(args.race):
            parser.error("--race-weights length must match --race length")
    
    if args.ethnicity_weights and args.ethnicity:
        if len(args.ethnicity_weights) != len(args.ethnicity):
            parser.error("--ethnicity-weights length must match --ethnicity length")

    # Step 1: Generate CSV
    print(f"🔄 Generating {args.rows} rows from schema {args.schema}...")
    df = generate_csv(args.schema, args.rows,
                     age_min=args.age_min, age_max=args.age_max,
                     gender_list=args.gender, gender_weights=args.gender_weights,
                     race_list=args.race, race_weights=args.race_weights,
                     ethnicity_list=args.ethnicity, ethnicity_weights=args.ethnicity_weights)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out, index=False)
    print(f" CSV generated → {args.csv_out}")

    # Step 2: Convert to HL7
    print(f" Converting CSV to HL7 v2.5.1 ADT messages...")
    messages = df_to_hl7_messages(df)
    args.hl7_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.hl7_out, 'w', encoding='utf-8', newline='') as f:
        for msg in messages:
            f.write(msg)
            f.write('\n')
    print(f" HL7 messages generated → {args.hl7_out}")
    print(f"\n Summary:")
    print(f"   - Rows:    {len(df)}")
    print(f"   - Columns: {len(df.columns)}")
    print(f"   - Columns with data: {sum(df[c].astype(str).ne('').any() for c in df.columns)}/{len(df.columns)}")
    print(f"   - Messages: {len(messages)}")


if __name__ == '__main__':
    main()
