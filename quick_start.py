"""
Quick Start - Run Analysis Without PyTorch
===========================================

This script runs the semantic term analysis without requiring PyTorch.
Perfect for getting immediate results for your report!

Usage:
    python quick_start.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*60)
    print("SEMANTIC MASTERING - QUICK START")
    print("="*60)

    print("\nThis will run semantic term analysis on SocialFX data.")
    print("No neural network training required!\n")

    # Check if we're in the right directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}\n")

    # Run semantic term analysis
    print("[1/1] Running Semantic Term Analysis...")
    print("-"*60)

    terms = ['warm', 'bright', 'punchy', 'smooth', 'heavy', 'soft', 'harsh']

    try:
        # Run the analysis
        result = subprocess.run([
            sys.executable,
            'research/semantic_term_analysis.py',
            '--terms', *terms,
            '--output', './semantic_analysis_results'
        ], check=True, capture_output=False, text=True)

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print("\nResults saved to: ./semantic_analysis_results/")
        print("\nGenerated files:")
        print("  - semantic_analysis_warm.png")
        print("  - semantic_analysis_bright.png")
        print("  - semantic_analysis_punchy.png")
        print("  - ... (one per term)")
        print("  - semantic_comparison_heatmap.png")
        print("  - semantic_analysis_report.md")
        print("\nUse these visualizations in your report!")

    except subprocess.CalledProcessError as e:
        print(f"\nError running analysis: {e}")
        print("\nTry running directly:")
        print("  cd research")
        print(f"  python semantic_term_analysis.py --terms {' '.join(terms)}")
    except FileNotFoundError:
        print("\nCouldn't find semantic_term_analysis.py")
        print("Make sure you're running this from the semantic_mastering_system directory")

if __name__ == '__main__':
    main()
