# ADT Message Evaluation Metrics

## Overview
The evaluation system scores ADT messages on a scale of 0-100% to measure quality and completeness against HL7 v2.5 standards.

## The Complete Quality Score Formula

```
Overall Score = (Required × 40%) + (Standard × 30%) + (Completeness × 30%)
```

## Components

### 1. Required Segments (40% weight)
Checks for  HL7 ADT segments:
- **MSH** - Message Header
- **EVN** - Event Type
- **PID** - Patient Identification
- **PV1** - Patient Visit

**Calculation:** (Present Required Segments / 4) × 100

### 2. Standard Segments (30% weight)
Checks for all 14 STANDARD HL7 ADT segment types:
- MSH, EVN, PID, NK1, PV1, PV2, OBX, AL1, DG1, PR1, ROL, GT1, IN1, IN2

**Calculation:** (Unique Segment Types Present / 14) × 100

### 3. Field Completeness (30% weight)
Measures how many fields contain actual data vs. empty fields.

**Calculation:** (Populated Fields / Total Fields) × 100

## Quality Tiers

Score , Rating , Description |
80-100% , ✓ EXCELLENT , Production ready, comprehensive data
70-79% , ○ GOOD , Acceptable quality, minor gaps
60-69% , ! FAIR , Basic structure, needs enrichment
<60% , ✗ NEEDS WORK , Missing critical segments/data 

## Example Calculation

**Message with:**
- Required segments: 4/4 present = 100%
- Standard segments: 10/14 types = 71.4%
- Field completeness: 55/100 fields = 55%

**Score:** (100 × 0.40) + (71.4 × 0.30) + (55 × 0.30) = **76.4% (GOOD)**
