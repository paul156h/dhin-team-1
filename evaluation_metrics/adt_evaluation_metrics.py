# This script compares generated ADT messages with a reference sample and provides
# similarity metrics including segment completeness, field presence, and data quality.

import os
import re
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# Reference ADT message from interfaceware.com
REFERENCE_ADT = r"""MSH|^~\&|MESA_ADT|XYZ_ADMITTING|iFW|ZYX_HOSPITAL|||ADT^A04|103102|P|2.4||||||||
EVN||200007010800||||200007010800
PID|||583295^^^ADT1||DOE^JANE||19610615|M-||2106-3|123 MAIN STREET^^GREENSBORO^NC^27401-1020|GL|(919)379-1212|(919)271-3434~(919)277-3114||S||PATID12345001^2^M10|123456789|9-87654^NC
NK1|1|BATES^RONALD^L|SPO|||||20011105
PV1||E||||||5101^NELL^FREDERICK^P^^DR|||||||||||V1295^^^ADT1|||||||||||||||||||||||||200007010800||||||||
PV2|||^ABDOMINAL PAIN
OBX|1|HD|SR Instance UID||1.123456.2.2000.31.2.1||||||F||||||
AL1|1||^PENICILLIN||PRODUCES HIVES~RASH
AL1|2||^CAT DANDER
DG1|001|I9|1550|MAL NEO LIVER, PRIMARY|19880501103005|F||
PR1|2234|M11|111^CODE151|COMMON PROCEDURES|198809081123
ROL|45^RECORDER^ROLE MASTER LIST|AD|CP|KATE^SMITH^ELLEN|199505011201
GT1|1122|1519|BILL^GATES^A
IN1|001|A357|1234|BCMD|||||132987
IN2|ID1551001|SSN12345678"""

# Standard ADT message segments
STANDARD_SEGMENTS = {
    'MSH': 'Message Header',
    'EVN': 'Event Type',
    'PID': 'Patient Identification',
    'NK1': 'Next of Kin',
    'PV1': 'Patient Visit',
    'PV2': 'Patient Visit Additional Info',
    'OBX': 'Observation/Result',
    'AL1': 'Allergy Information',
    'DG1': 'Diagnosis Information',
    'PR1': 'Procedures',
    'ROL': 'Role',
    'GT1': 'Guarantor Information',
    'IN1': 'Insurance Information',
    'IN2': 'Insurance Information Detail'
}

# Minimum required segments for a valid ADT message
REQUIRED_SEGMENTS = {'MSH', 'EVN', 'PID', 'PV1'}

# HL7 Table 0001 — Administrative Sex
VALID_GENDER_CODES = {'M', 'F', 'O', 'U', 'A', 'N', 'C'}

# HL7 Table 0005 / CDC OMB — Race coded identifiers and common text equivalents
VALID_RACE_CODES = {
    '1002-5', '2028-9', '2054-5', '2076-8', '2106-3', '2131-1', 'ASKU', 'UNK',
    'White', 'Black', 'Asian', 'Hispanic',
    'American Indian', 'Pacific Islander', 'Other', 'Two or More',
    'Black or African American', 'Two or more races',
    'Native Hawaiian or Other Pacific Islander',
    'American Indian or Alaska Native',
    'Unknown', 'Declined to state', 'Not Reported',
}

# HL7 Table 0189 / CDC PHIN — Ethnicity coded identifiers and text equivalents
VALID_ETHNICITY_CODES = {
    '2135-2', '2186-5', 'ASKU', 'UNK',
    'Hispanic or Latino', 'Not Hispanic or Latino',
    'Unknown', 'Declined to state', 'Not Reported',
}

SSN_PATTERN = re.compile(r'^\d{9}$|^\d{3}-\d{2}-\d{4}$')
PATIENT_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\-_\.]{0,49}$')
INSURANCE_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9\-_\./]{0,49}$')


@dataclass
class ContentValidationResult:
    checks_total: int
    checks_passed: int
    issues: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        if self.checks_total == 0:
            return 100.0
        return round((self.checks_passed / self.checks_total) * 100, 1)


@dataclass
class SegmentMetrics:

    segment_type: str
    present: bool
    expected_fields: int
    actual_fields: int
    field_completeness: float
    populated_fields: int
    
    def __post_init__(self):
        if self.actual_fields > 0:
            self.field_completeness = (self.populated_fields / self.actual_fields) * 100
        else:
            self.field_completeness = 0.0


@dataclass
class MessageMetrics:

    filename: str
    total_segments: int
    unique_segment_types: int
    required_segments_present: float
    standard_segments_present: float
    segment_completeness: float
    field_completeness: float
    overall_similarity: float
    missing_segments: List[str]
    segment_details: Dict[str, Dict]
    content_validation: ContentValidationResult = field(
        default_factory=lambda: ContentValidationResult(0, 0, [])
    )


class ADTEvaluator:
    def __init__(self, reference_message: str = REFERENCE_ADT):
        self.reference_message = reference_message
        self.reference_segments = self._parse_message(reference_message)
        
    def _parse_message(self, message: str) -> Dict[str, List[str]]:
        segments = defaultdict(list)
        for line in message.strip().split('\n'):
            if not line.strip():
                continue
            segment_type = line[:3]
            fields = line.split('|')[1:]
            segments[segment_type].append(fields)
        return segments

    def _count_populated_fields(self, fields: List[str]) -> int:
        return sum(1 for field in fields if field and field.strip())

    def _extract_field_value(self, field: str) -> str:
        if not field:
            return ""
        return field.split('^')[0].strip() if '^' in field else field.strip()
    
    def _validate_date(self, value: str, field_name: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        raw = value.strip()
        date_part = raw[:8]
        if not re.match(r'^\d{8}$', date_part):
            return False, f"{field_name}: '{raw}' is not in YYYYMMDD format"
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

    def _validate_gender(self, value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        code = value.strip()
        if code.upper() not in {c.upper() for c in VALID_GENDER_CODES}:
            valid_list = ', '.join(sorted(VALID_GENDER_CODES))
            return False, f"Gender '{code}' not in HL7 Table 0001 — valid: {valid_list}"
        return True, ''

    def _validate_race(self, value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        code = value.strip()
        if code not in VALID_RACE_CODES:
            return False, f"Race '{code}' not in HL7 Table 0005 / CDC OMB value set"
        return True, ''

    def _validate_ethnicity(self, value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        code = value.strip()
        if code not in VALID_ETHNICITY_CODES:
            return False, f"Ethnicity '{code}' not in HL7 Table 0189 / CDC PHIN value set"
        return True, ''

    def _validate_patient_id(self, value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        pid = value.split('^')[0].strip()
        if not pid:
            return True, ''
        if not PATIENT_ID_PATTERN.match(pid):
            return False, f"Patient ID '{pid}' contains invalid characters or exceeds 50 chars"
        return True, ''

    def _validate_ssn(self, value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        ssn = value.strip()
        if not SSN_PATTERN.match(ssn):
            return False, f"SSN '{ssn}' not in valid format (XXXXXXXXX or XXX-XX-XXXX)"
        return True, ''

    def _validate_insurance_id(self, value: str) -> Tuple[bool, str]:
        if not value or not value.strip():
            return True, ''
        ins_id = value.strip()
        if not INSURANCE_ID_PATTERN.match(ins_id):
            return False, f"Insurance ID '{ins_id}' contains invalid characters or format"
        return True, ''

    def _validate_message_content(self, message_segments: Dict) -> ContentValidationResult:
        issues: List[str] = []
        passed = 0
        total  = 0

        def run(ok: bool, issue: str) -> None:
            nonlocal passed, total
            total += 1
            if ok:
                passed += 1
            elif issue:
                issues.append(issue)

        def field(fields: List[str], idx: int) -> str:
            return fields[idx].strip() if len(fields) > idx else ''

        if 'PID' in message_segments:
            pid = message_segments['PID'][0]

            ok, msg = self._validate_patient_id(field(pid, 1))
            run(ok, f"PID.2: {msg}")

            ok, msg = self._validate_date(field(pid, 5), 'PID DOB')
            run(ok, msg)

            ok, msg = self._validate_gender(field(pid, 6))
            run(ok, f"PID.8 (gender): {msg}")

            race_code = field(pid, 9)
            if race_code and '^^' not in race_code and not re.match(r'^\d+\s', race_code):
                ok, msg = self._validate_race(race_code)
                run(ok, f"PID.10 (race code): {msg}")

            _race_text_values = {v for v in VALID_RACE_CODES
                                 if not re.match(r'^\d+-\d+$', v)
                                 and v not in ('ASKU', 'UNK')}
            _eth_text_values = {v for v in VALID_ETHNICITY_CODES
                                if not re.match(r'^\d+-\d+$', v)
                                and v not in ('ASKU', 'UNK')}

            found_race_text = ''
            found_ethnicity = ''
            for f_val in pid:
                v = f_val.strip()
                if v in _race_text_values:
                    found_race_text = v
                if v in _eth_text_values:
                    found_ethnicity = v

            if found_race_text:
                ok, msg = self._validate_race(found_race_text)
                run(ok, f"PID race text: {msg}")

            if found_ethnicity:
                ok, msg = self._validate_ethnicity(found_ethnicity)
                run(ok, f"PID ethnicity: {msg}")

        if 'MSH' in message_segments:
            msh = message_segments['MSH'][0]
            ok, msg = self._validate_date(field(msh, 5), 'MSH.7 (message timestamp)')
            run(ok, msg)

        if 'EVN' in message_segments:
            evn = message_segments['EVN'][0]
            ok, msg = self._validate_date(field(evn, 1), 'EVN.2 (recorded date)')
            run(ok, msg)

        if 'IN2' in message_segments:
            in2 = message_segments['IN2'][0]
            ok, msg = self._validate_ssn(field(in2, 1))
            run(ok, f"IN2.2: {msg}")

        if 'IN1' in message_segments:
            in1 = message_segments['IN1'][0]
            ok, msg = self._validate_insurance_id(field(in1, 2))
            run(ok, f"IN1.3: {msg}")

        return ContentValidationResult(
            checks_total=total,
            checks_passed=passed,
            issues=issues,
        )

    def evaluate_message(self, message_path: str) -> MessageMetrics:
        with open(message_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_text = f.read()

        message_segments = self._parse_message(message_text)
        filename = Path(message_path).name

        total_segments = sum(len(seg_list) for seg_list in message_segments.values())
        unique_segment_types = len(message_segments)

        required_present = sum(1 for seg in REQUIRED_SEGMENTS if seg in message_segments)
        required_completeness = (required_present / len(REQUIRED_SEGMENTS)) * 100

        standard_present = sum(1 for seg in STANDARD_SEGMENTS if seg in message_segments)
        standard_completeness = (standard_present / len(STANDARD_SEGMENTS)) * 100

        segment_details = {}
        total_expected_fields = 0
        total_populated_fields = 0
        
        for seg_type in STANDARD_SEGMENTS:
            if seg_type in message_segments:
                seg_instances = message_segments[seg_type]
                fields = seg_instances[0] if seg_instances else []
                
                expected_count = len(fields) if isinstance(fields, list) else 0
                populated_count = self._count_populated_fields(fields) if isinstance(fields, list) else 0
                
                total_expected_fields += expected_count
                total_populated_fields += populated_count
                
                segment_details[seg_type] = {
                    'present': True,
                    'count': len(seg_instances),
                    'expected_fields': expected_count,
                    'populated_fields': populated_count,
                    'completeness': (populated_count / expected_count * 100) if expected_count > 0 else 0
                }
            else:
                segment_details[seg_type] = {
                    'present': False,
                    'count': 0,
                    'expected_fields': 0,
                    'populated_fields': 0,
                    'completeness': 0
                }
        
        field_completeness = (total_populated_fields / total_expected_fields * 100) if total_expected_fields > 0 else 0
        segment_completeness = (unique_segment_types / len(STANDARD_SEGMENTS)) * 100

        content_validation = self._validate_message_content(message_segments)
        content_score = content_validation.score

        overall_similarity = (
            required_completeness * 0.35 +
            standard_completeness * 0.25 +
            field_completeness    * 0.25 +
            content_score         * 0.15
        )

        missing_segments = [seg for seg in REQUIRED_SEGMENTS if seg not in message_segments]

        return MessageMetrics(
            filename=filename,
            total_segments=total_segments,
            unique_segment_types=unique_segment_types,
            required_segments_present=required_completeness,
            standard_segments_present=standard_completeness,
            segment_completeness=segment_completeness,
            field_completeness=field_completeness,
            overall_similarity=overall_similarity,
            missing_segments=missing_segments,
            segment_details=segment_details,
            content_validation=content_validation,
        )
    
    def evaluate_directory(self, directory: str) -> List[MessageMetrics]:
        results = []
        adt_files = list(Path(directory).glob('*.adt.txt'))
        
        for adt_file in sorted(adt_files):
            try:
                metrics = self.evaluate_message(str(adt_file))
                results.append(metrics)
            except Exception as e:
                print(f"Error evaluating {adt_file}: {e}")
                continue
        
        return results
    
    def generate_report(self, results: List[MessageMetrics]) -> str:
        if not results:
            return "No ADT messages found to evaluate."

        report = []
        report.append("=" * 80)
        report.append("ADT MESSAGE EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        avg_required = sum(r.required_segments_present for r in results) / len(results)
        avg_standard = sum(r.standard_segments_present for r in results) / len(results)
        avg_field_completeness = sum(r.field_completeness for r in results) / len(results)

        if avg_similarity >= 80:
            quality_tier = "EXCELLENT"
        elif avg_similarity >= 70:
            quality_tier = "GOOD"
        elif avg_similarity >= 60:
            quality_tier = "FAIR"
        else:
            quality_tier = "NEEDS IMPROVEMENT"

        report.append("SUMMARY")
        report.append("-" * 80)
        report.append(f"Messages Evaluated:  {len(results)}")
        report.append(f"Average Quality:     {avg_similarity:.1f}%  ({quality_tier})")
        report.append("")

        report.append("DETAILED RESULTS")
        report.append("-" * 80)

        for idx, result in enumerate(results, 1):
            if result.overall_similarity >= 80:
                quality_icon = "+"
            elif result.overall_similarity >= 70:
                quality_icon = "~"
            else:
                quality_icon = "!"

            report.append(f"\n[{idx:2d}] {result.filename}")
            report.append(f"    Quality: {result.overall_similarity:.1f}% [{quality_icon}]")
            report.append(f"    Segments: {result.unique_segment_types}/14 types, {result.total_segments} total")
            report.append(f"    Field Data: {result.field_completeness:.1f}%")
            
            if result.missing_segments:
                missing = ', '.join(result.missing_segments)
                report.append(f"    Missing Required: {missing}")

            cv = result.content_validation
            report.append(f"    Content Validity: {cv.score:.1f}% ({cv.checks_passed}/{cv.checks_total} checks passed)")
            for issue in cv.issues:
                report.append(f"      - {issue}")

            present_segs = [seg for seg, det in result.segment_details.items() if det['present']]
            if present_segs:
                seg_list = ", ".join(f"{s}({result.segment_details[s]['count']})" for s in present_segs)
                report.append(f"    Present: {seg_list}")
        
        report.append("")
        all_issues = []
        for result in results:
            for issue in result.content_validation.issues:
                all_issues.append(f"  [{result.filename}] {issue}")

        if all_issues:
            report.append("CONTENT VALIDATION ISSUES")
            report.append("-" * 80)
            for issue in all_issues:
                report.append(issue)
            report.append("")

        report.append("SCORING GUIDE")
        report.append("-" * 80)
        report.append("Quality Score = Required (35%) + Standard (25%) + Completeness (25%) + Content Validity (15%)")
        report.append("")
        report.append("Content Validity checks:")
        report.append("  Dates (YYYYMMDD, plausible range), Gender (HL7 Table 0001: M/F/O/U/A/N/C)")
        report.append("  Race (HL7 Table 0005 / CDC OMB codes+text), Ethnicity (HL7 Table 0189)")
        report.append("  Patient IDs (alphanumeric), SSNs (9-digit or XXX-XX-XXXX), Insurance IDs")
        report.append("")
        report.append("Quality Tiers:")
        report.append("  [+] EXCELLENT (80-100%)  - Production ready, comprehensive data")
        report.append("  [~] GOOD (70-79%)        - Acceptable quality, minor gaps")
        report.append("  [!] FAIR (60-69%)        - Basic structure, needs enrichment")
        report.append("  [-] NEEDS WORK (<60%)    - Missing critical segments/data")
        report.append("")
        
        return "\n".join(report)


def main():
    import sys

    project_root = Path(__file__).parent.parent
    adt_dir = project_root / "test_data" / "adt_messages"
    outputs_dir = project_root / "outputs"

    if not adt_dir.exists():
        print(f"ADT messages directory not found: {adt_dir}")
        sys.exit(1)

    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ADTEvaluator()
    results = evaluator.evaluate_directory(str(adt_dir))

    report = evaluator.generate_report(results)
    print(report)

    report_path = outputs_dir / "adt_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")

    if results:
        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        print(f"Average similarity score: {avg_similarity:.2f}%")


if __name__ == "__main__":
    main()
