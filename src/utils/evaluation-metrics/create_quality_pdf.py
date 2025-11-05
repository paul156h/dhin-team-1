#!/usr/bin/env python3
"""
Simple PDF visualization of ADT evaluation metrics.
Creates histograms comparing different quality metrics.
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


def create_visualization_pdf(results: List[MessageMetrics], output_path: str):
    """Create a PDF with histogram visualizations of ADT metrics"""
    
    if not results:
        print("No results to visualize")
        return
    
    # Extract metrics for visualization
    similarity_scores = [r.overall_similarity for r in results]
    required_segments = [r.required_segments_present for r in results]
    standard_segments = [r.standard_segments_present for r in results]
    field_completeness = [r.field_completeness for r in results]
    total_segments = [r.total_segments for r in results]
    unique_segments = [r.unique_segment_types for r in results]
    
    # Create PDF
    with pdf_backend.PdfPages(output_path) as pdf:
        # Page 1: Main Quality Metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('ADT Message Quality Metrics', fontsize=16, fontweight='bold')
        
        # Overall Similarity Score
        ax1.hist(similarity_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Overall Similarity Score (%)')
        ax1.set_ylabel('Number of Messages')
        ax1.set_title('Overall Similarity Distribution')
        ax1.axvline(np.mean(similarity_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(similarity_scores):.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Required Segments
        ax2.hist(required_segments, bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Required Segments Present (%)')
        ax2.set_ylabel('Number of Messages')
        ax2.set_title('Required Segments Completeness')
        ax2.axvline(np.mean(required_segments), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(required_segments):.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Standard Segments
        ax3.hist(standard_segments, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('Standard Segments Present (%)')
        ax3.set_ylabel('Number of Messages')
        ax3.set_title('Standard Segments Coverage')
        ax3.axvline(np.mean(standard_segments), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(standard_segments):.1f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Field Completeness
        ax4.hist(field_completeness, bins=10, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_xlabel('Field Completeness (%)')
        ax4.set_ylabel('Number of Messages')
        ax4.set_title('Data Completeness Distribution')
        ax4.axvline(np.mean(field_completeness), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(field_completeness):.1f}%')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Structure Metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('ADT Message Structure Metrics', fontsize=16, fontweight='bold')
        
        # Total Segments
        ax1.hist(total_segments, bins=8, alpha=0.7, color='plum', edgecolor='black')
        ax1.set_xlabel('Total Segments per Message')
        ax1.set_ylabel('Number of Messages')
        ax1.set_title('Message Size Distribution')
        ax1.axvline(np.mean(total_segments), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(total_segments):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Unique Segment Types
        ax2.hist(unique_segments, bins=8, alpha=0.7, color='lightsteelblue', edgecolor='black')
        ax2.set_xlabel('Unique Segment Types per Message')
        ax2.set_ylabel('Number of Messages')
        ax2.set_title('Message Complexity Distribution')
        ax2.axvline(np.mean(unique_segments), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(unique_segments):.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Summary Statistics
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('ADT Quality Summary Statistics', fontsize=16, fontweight='bold')
        
        # Create a summary table
        metrics_data = {
            'Overall Similarity': similarity_scores,
            'Required Segments': required_segments,
            'Standard Segments': standard_segments,
            'Field Completeness': field_completeness,
            'Total Segments': total_segments,
            'Unique Segments': unique_segments
        }
        
        stats_text = []
        stats_text.append("SYNTHETIC ADT MESSAGE QUALITY ANALYSIS")
        stats_text.append("=" * 50)
        stats_text.append(f"Total Messages Analyzed: {len(results)}")
        stats_text.append("")
        
        for metric_name, values in metrics_data.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            stats_text.append(f"{metric_name}:")
            stats_text.append(f"  Mean: {mean_val:.2f}")
            stats_text.append(f"  Std:  {std_val:.2f}")
            stats_text.append(f"  Min:  {min_val:.2f}")
            stats_text.append(f"  Max:  {max_val:.2f}")
            stats_text.append("")
        
        # Quality Assessment
        avg_similarity = np.mean(similarity_scores)
        stats_text.append("QUALITY ASSESSMENT:")
        stats_text.append("-" * 30)
        
        if avg_similarity >= 80:
            quality = "EXCELLENT"
        elif avg_similarity >= 70:
            quality = "GOOD"
        elif avg_similarity >= 60:
            quality = "FAIR"
        else:
            quality = "NEEDS IMPROVEMENT"
            
        stats_text.append(f"Overall Quality Rating: {quality}")
        stats_text.append(f"Average Similarity Score: {avg_similarity:.1f}%")
        stats_text.append("")
        stats_text.append("Recommendations:")
        
        if np.mean(required_segments) < 100:
            stats_text.append("• Ensure all required segments (MSH, EVN, PID, PV1) are present")
        if np.mean(field_completeness) < 70:
            stats_text.append("• Improve data completeness in message fields")
        if np.mean(standard_segments) < 50:
            stats_text.append("• Consider adding more standard HL7 segments")
        
        # Display text on the plot
        ax.text(0.05, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function"""
    # Get the outputs directory
    project_root = Path(__file__).parent.parent.parent.parent
    outputs_dir = project_root / "outputs"
    
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        print("Please run the message generator first to create ADT files.")
        sys.exit(1)
    
    # Check for ADT files
    adt_files = list(outputs_dir.glob('*.adt.txt'))
    if not adt_files:
        print(f"No ADT files found in {outputs_dir}")
        print("Please run the message generator first to create ADT files.")
        sys.exit(1)
    
    print(f"Found {len(adt_files)} ADT files to analyze...")
    
    # Evaluate messages
    evaluator = ADTEvaluator()
    results = evaluator.evaluate_directory(str(outputs_dir))
    
    if not results:
        print("No valid ADT messages could be processed.")
        sys.exit(1)
    
    # Create visualization
    pdf_path = outputs_dir / "adt_quality_visualization.pdf"
    print(f"Creating quality visualization PDF...")
    
    try:
        create_visualization_pdf(results, str(pdf_path))
        print(f"✓ Visualization saved to: {pdf_path}")
        
        # Print quick summary
        avg_similarity = sum(r.overall_similarity for r in results) / len(results)
        print(f"\n📊 Quick Summary:")
        print(f"   Messages analyzed: {len(results)}")
        print(f"   Average quality score: {avg_similarity:.1f}%")
        
        if avg_similarity >= 70:
            print(f"   ✅ Quality assessment: Good synthetic data!")
        else:
            print(f"   ⚠️  Quality assessment: Consider improvements")
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()