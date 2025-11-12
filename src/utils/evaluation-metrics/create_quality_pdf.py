#!/usr/bin/env python3
"""
Simplified PDF visualization of ADT evaluation metrics.
Clean, simple design with quality scorecard, structure comparison, and recommendations.
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
from typing import List

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from adt_evaluation_metrics import ADTEvaluator, MessageMetrics


def create_simplified_visualization_pdf(results: List[MessageMetrics], output_path: str):
    """Create a simplified PDF with clean visualizations"""
    
    if not results:
        print("No results to visualize")
        return
    
    # Extract metrics
    similarity_scores = [r.overall_similarity for r in results]
    total_segments = [r.total_segments for r in results]
    unique_segments = [r.unique_segment_types for r in results]
    filenames = [r.filename.replace('.adt.txt', '') for r in results]
    
    # Create PDF
    with pdf_backend.PdfPages(output_path) as pdf:
        
        # PAGE 1: QUALITY SCORECARD
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('ADT Message Quality Assessment', fontsize=16, fontweight='bold')
        
        # Sort by score
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_names = [filenames[i] for i in sorted_indices]
        sorted_scores = [similarity_scores[i] for i in sorted_indices]
        
        # Color coding
        colors = []
        for score in sorted_scores:
            if score >= 80:
                colors.append('#2ecc71')  # Green
            elif score >= 70:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        ax = fig.add_subplot(111)
        y_pos = np.arange(len(sorted_names))
        
        # Bar chart
        bars = ax.barh(y_pos, sorted_scores, color=colors, edgecolor='black', linewidth=1.5, height=0.55)
        
        # Labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            if score >= 80:
                rating = "EXCELLENT"
            elif score >= 70:
                rating = "GOOD"
            else:
                rating = "FAIR"
            ax.text(score + 1, bar.get_y() + bar.get_height()/2, 
                   f'{score:.1f}% - {rating}', va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Quality Score (%)', fontweight='bold', fontsize=9)
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)
        
        # Statistics - moved to bottom as plain text
        excellent = sum(1 for s in similarity_scores if s >= 80)
        good = sum(1 for s in similarity_scores if 70 <= s < 80)
        fair = sum(1 for s in similarity_scores if s < 70)
        
        stats_text = f"Total Messages: {len(results)}  |  Average: {np.mean(similarity_scores):.1f}%  |  Range: {np.min(similarity_scores):.1f}% - {np.max(similarity_scores):.1f}%  |  Excellent: {excellent}  |  Good: {good}  |  Fair: {fair}"
        
        ax.text(0.5, -0.08, stats_text, transform=ax.transAxes, fontsize=8.5, ha='center',
               style='italic', color='gray')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 2: MESSAGE STRUCTURE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Message Structure Comparison', fontsize=14, fontweight='bold')
        
        # Total Segments
        sorted_idx_seg = np.argsort(total_segments)[::-1]
        names_seg = [filenames[i] for i in sorted_idx_seg]
        vals_seg = [total_segments[i] for i in sorted_idx_seg]
        cols_seg = ['#2ecc71' if v >= 15 else '#f39c12' if v >= 10 else '#e74c3c' for v in vals_seg]
        
        y_pos = np.arange(len(names_seg))
        ax1.barh(y_pos, vals_seg, color=cols_seg, edgecolor='black', linewidth=1, height=0.6)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names_seg, fontsize=8)
        ax1.set_xlabel('Total Segments', fontweight='bold', fontsize=8)
        ax1.set_title('Message Size', fontweight='bold', fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        for i, v in enumerate(vals_seg):
            ax1.text(v + 0.3, i, str(int(v)), va='center', fontweight='bold')
        
        # Unique Segment Types
        sorted_idx_uniq = np.argsort(unique_segments)[::-1]
        names_uniq = [filenames[i] for i in sorted_idx_uniq]
        vals_uniq = [unique_segments[i] for i in sorted_idx_uniq]
        cols_uniq = ['#2ecc71' if v == 14 else '#f39c12' if v >= 12 else '#e74c3c' for v in vals_uniq]
        
        y_pos = np.arange(len(names_uniq))
        ax2.barh(y_pos, vals_uniq, color=cols_uniq, edgecolor='black', linewidth=1, height=0.6)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names_uniq, fontsize=8)
        ax2.set_xlabel('Unique Types', fontweight='bold', fontsize=8)
        ax2.set_title('Message Complexity (out of 14)', fontweight='bold', fontsize=9)
        ax2.set_xlim(0, 15)
        ax2.grid(axis='x', alpha=0.3)
        for i, v in enumerate(vals_uniq):
            ax2.text(v + 0.2, i, f'{int(v)}/14', va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 3: RECOMMENDATIONS
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.suptitle('Quality Analysis & Recommendations', fontsize=14, fontweight='bold')
        
        rec_text = f"""QUALITY SCORE INTERPRETATION
{'='*55}

OVERALL SIMILARITY SCORE
  Combines: Required segments (40%) + Standard segments (30%) 
            + Field completeness (30%)

CURRENT RESULTS
  Highest:  {sorted_names[0]} at {sorted_scores[0]:.1f}%
  Lowest:   {sorted_names[-1]} at {sorted_scores[-1]:.1f}%
  Average:  {np.mean(similarity_scores):.1f}%


QUALITY TIERS
  GREEN (EXCELLENT)   80%+  {excellent} message(s) - Production ready
  ORANGE (GOOD)       70-80% {good} message(s) - Good quality  
  RED (FAIR)          <70%  {fair} message(s) - Needs work


MESSAGE STRUCTURE
  Total Segments: {int(np.min(total_segments))}-{int(np.max(total_segments))} (more = richer data)
  Unique Types:   {int(np.min(unique_segments))}-{int(np.max(unique_segments))}/14 (14 = complete)


HOW TO IMPROVE
  1. Add missing segments (NK1, PV2, DG1, PR1, IN2, ROL)
  
  2. Populate more fields in existing segments
     (demographics, clinical notes, vital signs, diagnoses)
  
  3. Include multiple instances for rich data
     (multiple OBX for different observations,
      multiple DG1 for comorbidities, etc.)
  
  4. Target 14/14 segment types for maximum quality scores

"""
        
        ax.text(0.05, 0.95, rec_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function"""
    project_root = Path(__file__).parent.parent.parent.parent
    adt_dir = project_root / "data" / "adt_messages"
    outputs_dir = project_root / "outputs"
    
    if not adt_dir.exists():
        print(f"ADT messages directory not found: {adt_dir}")
        sys.exit(1)
    
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    adt_files = list(adt_dir.glob('*.adt.txt'))
    if not adt_files:
        print(f"No ADT files found in {adt_dir}")
        sys.exit(1)
    
    print(f"Found {len(adt_files)} ADT files to analyze...")
    
    evaluator = ADTEvaluator()
    results = evaluator.evaluate_directory(str(adt_dir))
    
    if not results:
        print("No valid ADT messages could be processed.")
        sys.exit(1)
    
    pdf_path = outputs_dir / "adt_quality_visualization.pdf"
    print(f"Creating simplified quality visualization PDF...")
    
    try:
        create_simplified_visualization_pdf(results, str(pdf_path))
        print(f"✓ Visualization saved to: {pdf_path}")
        
        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        print(f"\n📊 Summary: {len(results)} messages, {avg_similarity:.1f}% average quality")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
