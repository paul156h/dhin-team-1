import pandas as pd, random, string, uuid, re
from datetime import datetime, timedelta
from pathlib import Path

N_ROWS = 200  # <-- change this number to control how many lines get created

# Load your existing CSV to copy column names and order
src_path = Path("mimic-iv-demo-dataset.csv")
example_df = pd.read_csv(src_path, nrows=5)
columns = list(example_df.columns)

# Lookup tables
# Proof of concept for now, this can be changed in the future
SEX = ["M", "F"]
RACE_DESC = [
    "White", "Black or African American", "Asian",
    "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Other",
]
ADMISSION_TYPES = ["EMERGENCY", "ELECTIVE", "URGENT", "NEWBORN"]
ADMISSION_LOCS = ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "CLINIC REFERRAL", "TRANSFER FROM HOSPITAL", "WALK-IN"]
DISCHARGE_LOCS = ["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB", "EXPIRED"]
INSURANCE = ["Medicare", "Medicaid", "Private", "Self Pay"]
LANG = ["ENGL", "SPAN", "MAND", "VIET", "ARAB"]
RELIGION = ["CATHOLIC", "PROTESTANT QUAKER", "JEWISH", "BUDDHIST", "HINDU", "MUSLIM", "CHRISTIAN", "OTHER"]
MARITAL = ["MARRIED", "SINGLE", "DIVORCED", "WIDOWED"]
CITIES = [
    ("Greensboro","NC","27401"), ("Raleigh","NC","27601"), ("Durham","NC","27701"),
    ("Seattle","WA","98101"), ("Boston","MA","02108"), ("Atlanta","GA","30303"),
    ("Austin","TX","78701"), ("New York","NY","10001"), ("Chicago","IL","60601"),
]
FAMILY_NAMES = ["DOE","SMITH","JOHNSON","BROWN","JONES","MILLER","DAVIS","GARCIA","RODRIGUEZ","WILSON"]
GIVEN_NAMES  = ["JANE","JOHN","EMMA","OLIVIA","NOAH","LIAM","AVA","ISABELLA","MIA","ELIJAH"]
ICD10_COMMON = [
    ("R10.9", "Unspecified abdominal pain"), ("J06.9", "Acute upper respiratory infection, unspecified"),
    ("M54.5", "Low back pain"), ("I10", "Essential (primary) hypertension"),
    ("E11.9", "Type 2 diabetes mellitus without complications"), ("R07.9", "Chest pain, unspecified"),
]

# Helper functions
def rand_digits(n): return "".join(random.choices(string.digits, k=n))
def rand_phone(): return f"({rand_digits(3)}){rand_digits(3)}-{rand_digits(4)}"
def rand_street():
    return f"{random.randint(10,9999)} {random.choice(['MAIN ST','OAK AVE','PINE ST','ELM ST','MAPLE AVE','2ND ST'])}"
def rand_datetime(start_year=2015, end_year=2025):
    start, end = datetime(start_year,1,1), datetime(end_year,10,31)
    return start + timedelta(seconds=random.randint(0, int((end-start).total_seconds())))
def rand_dob(min_age=0, max_age=95):
    today = datetime(2025, 11, 6)
    age = random.randint(min_age, max_age)
    birth = today - timedelta(days=age*365 + random.randint(0,364))
    return birth
def normalize(name): return re.sub(r'[^a-z0-9]+', '', name.lower())

# Row generator that fills known columns based on their names
def make_row(columns):
    subj_id = int(rand_digits(7))
    hadm_id = int(rand_digits(7))
    adm_time = rand_datetime()
    disch_time = adm_time + timedelta(hours=random.randint(6, 240))
    expire_flag = 1 if random.random() < 0.03 else 0

    fam = random.choice(FAMILY_NAMES)
    giv = random.choice(GIVEN_NAMES)
    dob = rand_dob(0, 95)
    age_years = int((adm_time.date() - dob.date()).days // 365)
    race = random.choice(RACE_DESC)
    sex = random.choice(SEX)
    city, state, zipcode = random.choice(CITIES)
    street = rand_street()
    phone = rand_phone()
    dx_code, dx_desc = random.choice(ICD10_COMMON)
    admission_type = random.choice(ADMISSION_TYPES)
    admission_loc = random.choice(ADMISSION_LOCS)
    discharge_loc = "EXPIRED" if expire_flag else random.choice([x for x in DISCHARGE_LOCS if x != "EXPIRED"])
    insurance = random.choice(INSURANCE)
    language = random.choice(LANG)
    religion = random.choice(RELIGION)
    marital = random.choice(MARITAL)

    row = {}
    for col in columns:
        key = normalize(col)
        val = ""

        if key in ("subject_id","patientid","pid3"): val = subj_id
        elif key in ("hadm_id","visitid","pv1visitnumber"): val = hadm_id
        elif key in ("lastname","familyname"): val = fam.title()
        elif key in ("firstname","givenname"): val = giv.title()
        elif key in ("fullname","patientname"): val = f"{fam}^{giv}"
        elif key in ("gender","sex"): val = sex
        elif key in ("race","racedesc"): val = race
        elif key in ("dob","birthdate"): val = dob.strftime("%Y-%m-%d")
        elif key in ("age","ageyears"): val = age_years
        elif key in ("street","address"): val = street.title()
        elif key in ("city",): val = city
        elif key in ("state",): val = state
        elif key in ("zip","zipcode"): val = zipcode
        elif key in ("phone",): val = phone
        elif key in ("admittime","admissiontime"): val = adm_time.strftime("%Y-%m-%d %H:%M:%S")
        elif key in ("dischtime","dischargetime"): val = disch_time.strftime("%Y-%m-%d %H:%M:%S")
        elif key in ("admissiontype",): val = admission_type
        elif key in ("admissionlocation",): val = admission_loc
        elif key in ("dischargelocation",): val = discharge_loc
        elif key in ("insurance",): val = insurance
        elif key in ("language",): val = language
        elif key in ("religion",): val = religion
        elif key in ("maritalstatus",): val = marital
        elif key in ("diagnosis","dxdesc"): val = dx_desc
        elif key in ("dx","diagnosiscode"): val = dx_code
        elif key in ("hospitalexpireflag","deathflag"): val = expire_flag
        elif key in ("messagetype","msh9messagetype"): val = "ADT^A04"
        elif key in ("processingid","msh11processingid"): val = "P"
        elif key in ("versionid","msh12versionid"): val = "2.5.1"

        row[col] = val
    return row

# Generate all rows and save CSV
rows = [make_row(columns) for _ in range(N_ROWS)]
out_df = pd.DataFrame(rows, columns=columns)
out_df.to_csv("./generated_csvs/generated_data.csv", index=False)

print(f"✅ Generated {N_ROWS} rows -> generated_csvs/generated_data.csv")
