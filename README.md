generate_synthetic_adt.py — README

## What this script does

`generate_synthetic_adt.py` is an all-in-one synthetic ADT generator for testing HL7 v2.5.1 pipelines. Given a CSV schema (column names + order) it:

- Synthesizes realistic patient/visit fields (demographics, identifiers, timestamps, providers, insurance, diagnosis).
- Produces a CSV file matching the input schema with the generated rows.
- Converts each CSV row into an HL7 v2.5.1 ADT message (MSH, EVN, PID, PV1, DG1, IN1, NK1 segments) and writes them into a single `.hl7` file.

The generator is intentionally schema-flexible: it will only populate columns that exist in the provided schema, so you can supply a minimal or a full HL7-like schema depending on how much detail you want to generate.

## Quick start / prerequisites

- Python 3.8+ (tested with Python 3.9+)
- `pandas` library

Install dependencies (PowerShell):

```powershell
python -m pip install pandas
```

## Usage

Basic:

```powershell
python generate_synthetic_adt.py --schema schema_hl7_minimal.csv
```

Specify number of rows, seed and outputs:

```powershell
python generate_synthetic_adt.py --schema schema_hl7_minimal.csv --rows 100 --seed 42 --csv-out delaware_patients.csv --hl7-out delaware_adt.hl7
```

Arguments:

- `--schema` (required): Path to a CSV file used as the column-name/order template. The script reads the header row and generates those columns in order.
- `--rows` (default 200): Number of rows to generate.
- `--seed` (default 123): Deterministic seed for reproducible data.
- `--csv-out` (default `generated_adt.csv`): Output CSV path.
- `--hl7-out` (default `generated_adt_messages.hl7`): Output HL7 file path.

All file paths are relative to your current working directory unless given as absolute paths.

## Example schema

A minimal HL7-oriented schema (included as `schema_hl7_minimal.csv`) looks like:

```
msh-3,msh-4,msh-5,msh-6,msh-7,msh-9,msh-10,evn-1,evn-2,subject_id,pid-5,pid-7,gender,race,ethnicity,pid-11,pid-13,pv1-2,pv1-3,pv1-7,pv1-10,pv1-19,pv1-44,pv1-45,diagnosiscode,diagnosis,insurance,in1,nk1
```

You can also provide a trimmed schema (for testing) such as `test_schema_delaware.csv` which contains only a few columns — the script will fill only the available columns.

## What fields are synthesized

The generator produces a variety of synthetic values, including (but not limited to):

- Identifiers: `subject_id` (PID-3) and `hadm_id` (PV1-19) formatted with an assigning authority
- Names: plausible first/last names from small public lists (PID-5)
- DOB derived from age with small jitter (PID-7)
- Sex, race, ethnicity (mapped to HL7 CE-like codes)
- Address and phone (synthetic Delaware ZIPs and `302-555-XXXX` phones)
- Admission/admit/discharge timestamps (MSH-7, EVN-2, PV1-44/45)
- Providers: simple NPI-like ID + name for attending/referring physicians
- Diagnosis (DG1) using a small ICD-10 lookup
- Insurance (IN1) with a generated policy number
- NK1 (Next of kin) simple placeholder

All fields are intentionally synthetic and designed for test/dev use only.

## HL7 output details

- Messages are assembled as HL7 v2.5.1 string segments separated with CR (`\r`) and written sequentially to the output `.hl7` file.
- The script builds these segments: `MSH`, `EVN`, `PID`, `PV1`, optional `DG1`, `IN1`, `NK1` when data/columns exist.
- Timestamps are formatted as `YYYYMMDDHHMMSS` (UTC naive). If you need timezone offsets or other formats, this can be added.

## Reproducibility & customization

- Use `--seed` for deterministic output.
- You can modify the small lookup lists in the script (names, providers, ICD10 list, zip codes, etc.) to better match your test scenario.
- The generator only fills columns present in the schema header, so you can expand or shrink the schema to control payload richness.

## Limitations and next steps

- This is a synthetic generator for development/testing; do not use produced data for any real patient workflows.
- The HL7 messages are syntactic (string segments) but not validated against a specific implementation guide or profile.
- Suggested improvements if you need more realism:
  - Add timezone-aware timestamps
  - Use larger, realistic name/address/provider lists or link to a small fixture database
  - Emit stricter HL7 field indexing for PID-3 subcomponents (ID^assigning authority^identifier type)
  - Add PV2/IN2/IN3 fields or support ADT^A01/A08 trigger variations
  - Provide an option to emit one HL7 file per message or to stream via MLLP

## Files in this repo

- `generate_synthetic_adt.py` — combined generator you run.
- `generate_adt_csv.py` — original CSV-only generator (kept for reference).
- `generate_adt_hl7.py` — CSV → HL7 converter (kept for reference).
- `schema_hl7_minimal.csv` — example HL7 schema used for testing.
- `test_schema_delaware.csv` — tiny schema used for quick tests.

## Evaluation Metrics

For analysis and evaluation of HL7 message quality and pipeline performance, see the **[evaluation-metrics branch](https://github.com/paul156h/dhin-team-1/tree/evaluation-metrics)**. This branch contains:

- Metrics for assessing HL7 message quality
- Pipeline performance evaluation tools
- Statistical analysis of generated data

## License & attribution

See the project `LICENSE` file in the repository root for license terms.

