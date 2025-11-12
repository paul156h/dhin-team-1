# This script compares generated ADT messages with a reference sample and provides
# similarity metrics including segment completeness, field presence, and data quality.

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
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


@dataclass
class SegmentMetrics:
    """Metrics for a single segment"""
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
    """Overall metrics for an ADT message"""
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


class ADTEvaluator:
    """Evaluates ADT message quality and similarity"""
    
    def __init__(self, reference_message: str = REFERENCE_ADT):
        """Initialize with a reference ADT message"""
        self.reference_message = reference_message
        self.reference_segments = self._parse_message(reference_message)
        
    def _parse_message(self, message: str) -> Dict[str, List[str]]:
        """Parse an ADT message into segments and fields"""
        segments = defaultdict(list)
        lines = message.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            # Extract segment type (first 3 characters)
            segment_type = line[:3]
            # Split fields by pipe (|)
            fields = line.split('|')[1:]  # Skip segment type
            segments[segment_type].append(fields)
            
        return segments
    
    def _count_populated_fields(self, fields: List[str]) -> int:
        """Count non-empty fields in a segment"""
        return sum(1 for field in fields if field and field.strip())
    
    def _extract_field_value(self, field: str) -> str:
        """Extract the base value from a field (before any subcomponents)"""
        if not field:
            return ""
        # Remove subcomponents (^) to get base value
        return field.split('^')[0].strip() if '^' in field else field.strip()
    
    def evaluate_message(self, message_path: str) -> MessageMetrics:
        """Evaluate a single ADT message file"""
        with open(message_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_text = f.read()
        
        message_segments = self._parse_message(message_text)
        filename = Path(message_path).name
        
        # Calculate metrics
        total_segments = sum(len(seg_list) for seg_list in message_segments.values())
        unique_segment_types = len(message_segments)
        
        # Check for required segments
        required_present = sum(1 for seg in REQUIRED_SEGMENTS if seg in message_segments)
        required_completeness = (required_present / len(REQUIRED_SEGMENTS)) * 100
        
        # Check for standard segments
        standard_present = sum(1 for seg in STANDARD_SEGMENTS if seg in message_segments)
        standard_completeness = (standard_present / len(STANDARD_SEGMENTS)) * 100
        
        # Calculate field completeness
        segment_details = {}
        total_expected_fields = 0
        total_populated_fields = 0
        
        for seg_type in STANDARD_SEGMENTS:
            if seg_type in message_segments:
                seg_instances = message_segments[seg_type]
                # Use the first instance of the segment as representative
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
        
        # Calculate overall similarity
        # Weight: 40% required segments, 30% standard segments, 30% field completeness
        overall_similarity = (
            required_completeness * 0.4 +
            standard_completeness * 0.3 +
            field_completeness * 0.3
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
            segment_details=segment_details
        )
    
    def evaluate_directory(self, directory: str) -> List[MessageMetrics]:
        """Evaluate all ADT files in a directory"""
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
        """Generate a detailed evaluation report"""
        if not results:
            return "No ADT messages found to evaluate."
        
        report = []
        report.append("=" * 80)
        report.append("ADT MESSAGE EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        avg_required = sum(r.required_segments_present for r in results) / len(results)
        avg_standard = sum(r.standard_segments_present for r in results) / len(results)
        avg_field_completeness = sum(r.field_completeness for r in results) / len(results)
        
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append(f"Total messages evaluated: {len(results)}")
        report.append(f"Average overall similarity: {avg_similarity:.2f}%")
        report.append(f"Average required segments present: {avg_required:.2f}%")
        report.append(f"Average standard segments present: {avg_standard:.2f}%")
        report.append(f"Average field completeness: {avg_field_completeness:.2f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 80)
        
        for result in results:
            report.append(f"\nFile: {result.filename}")
            report.append(f"  Overall Similarity: {result.overall_similarity:.2f}%")
            report.append(f"  Total Segments: {result.total_segments}")
            report.append(f"  Unique Segment Types: {result.unique_segment_types}/{len(STANDARD_SEGMENTS)}")
            report.append(f"  Required Segments Present: {result.required_segments_present:.2f}%")
            report.append(f"  Standard Segments Present: {result.standard_segments_present:.2f}%")
            report.append(f"  Field Completeness: {result.field_completeness:.2f}%")
            
            if result.missing_segments:
                report.append(f"  [WARNING] Missing required segments: {', '.join(result.missing_segments)}")
            
            # Segment-level details
            report.append("  Segment Details:")
            for seg_type, details in result.segment_details.items():
                if details['present']:
                    report.append(f"    {seg_type:4} ({STANDARD_SEGMENTS[seg_type]:30}): "
                                f"Count={details['count']}, Fields populated={details['populated_fields']}/{details['expected_fields']}, "
                                f"Completeness={details['completeness']:.1f}%")
                else:
                    report.append(f"    {seg_type:4} ({STANDARD_SEGMENTS[seg_type]:30}): NOT PRESENT")
        
        report.append("\n" + "=" * 80)
        report.append("EVALUATION METRICS EXPLANATION")
        report.append("=" * 80)
        report.append("""
Overall Similarity Score:
- Weighted average of three metrics:
  * Required Segments Present (40% weight): MSH, EVN, PID, PV1
  * Standard Segments Present (30% weight): All standard HL7 ADT segments
  * Field Completeness (30% weight): Percentage of fields with actual data

Segment Presence:
- Required segments ensure minimum valid ADT structure
- Standard segments check compliance with full ADT specification

Field Completeness:
- Percentage of non-empty fields in present segments
- Higher values indicate more detailed patient information

A score above 70% generally indicates good quality ADT messages.
""")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    import sys
    
    # Get the data directory with ADT messages
    project_root = Path(__file__).parent.parent.parent.parent
    adt_dir = project_root / "data" / "adt_messages"
    outputs_dir = project_root / "outputs"
    
    if not adt_dir.exists():
        print(f"ADT messages directory not found: {adt_dir}")
        sys.exit(1)
    
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    # Evaluate messages
    evaluator = ADTEvaluator()
    results = evaluator.evaluate_directory(str(adt_dir))
    
    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save report to file
    report_path = outputs_dir / "adt_evaluation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Return summary statistics
    if results:
        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        print(f"\n[SUMMARY] Average similarity score: {avg_similarity:.2f}%")


if __name__ == "__main__":
    main()
