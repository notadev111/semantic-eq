"""
Semantic Term Analysis: What do engineers REALLY mean?
========================================================

Comprehensive analysis of semantic terms in the SocialFX dataset:
1. Cluster analysis within each term (variance/consistency)
2. Cross-term comparison (how different are "warm" vs "bright"?)
3. Frequency response visualization for each term
4. Statistical insights for report writing

This answers: "What does 'warm' actually mean in terms of EQ adjustments?"

Usage:
    python semantic_term_analysis.py --terms warm bright punchy --output ./analysis_results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Import base system (no torch needed for analysis)
import sys
sys.path.insert(0, '..')

# Simple data loader - no dependencies on dasp or torch
class SimpleSocialFXLoader:
    """Lightweight loader for SocialFX data"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.dataset = []
        self.column_mapping = {'descriptor': 'text', 'parameters': 'param_values'}
        self.df_eq = None

    def load_dataset(self):
        """Load SocialFX dataset from HuggingFace"""
        print("Loading SocialFX dataset...")
        self.df_eq = pd.read_parquet("hf://datasets/seungheondoh/socialfx-original/data/eq-00000-of-00001.parquet")

        # Normalize text column
        self.df_eq['text'] = self.df_eq['text'].str.lower().str.strip()

        self.dataset = []
        for idx, row in self.df_eq.iterrows():
            param_keys = row['param_keys']
            param_values = row['param_values']

            eq_dict = {
                'semantic_term': str(row['text']),
                'text': str(row['text']),
                'parameters': dict(zip(param_keys, param_values)),
                'id': row['id']
            }
            self.dataset.append(eq_dict)

        print(f"Loaded {len(self.dataset)} EQ examples")
        print(f"Unique terms: {len(set(ex['semantic_term'] for ex in self.dataset))}")

        return self.dataset

    def _extract_params(self, param_values):
        """Extract parameter values as numpy array"""
        if isinstance(param_values, (list, np.ndarray)):
            return np.array(param_values, dtype=float)
        return None

SocialFXDataLoader = SimpleSocialFXLoader


class SemanticTermAnalyzer:
    """
    Analyze what semantic terms actually mean in EQ parameter space
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.loader = SocialFXDataLoader(cache_dir=cache_dir)
        self.loader.load_dataset()

        # EQ band frequency centers (for visualization)
        self.band_frequencies = [60, 200, 800, 3000, 8000, 16000]
        self.band_names = ['Sub Bass', 'Bass', 'Low-Mid', 'Mid', 'High-Mid', 'Treble']

    def get_term_examples(self, term: str) -> np.ndarray:
        """Get all EQ examples for a specific term"""

        desc_col = self.loader.column_mapping.get('descriptor', 'text')
        param_col = self.loader.column_mapping.get('parameters', 'param_values')

        term_data = self.loader.df_eq[self.loader.df_eq[desc_col] == term]

        examples = []
        for _, row in term_data.iterrows():
            try:
                params = self.loader._extract_params(row[param_col])
                if params is not None and len(params) > 0:
                    examples.append(params)
            except Exception:
                continue

        return np.array(examples) if examples else np.array([])

    def analyze_term_consistency(self, term: str) -> Dict:
        """
        Analyze how consistent engineers are when using a specific term

        Returns metrics about variance, clustering, etc.
        """

        examples = self.get_term_examples(term)

        if len(examples) == 0:
            return {
                'term': term,
                'n_examples': 0,
                'error': 'No examples found'
            }

        analysis = {
            'term': term,
            'n_examples': len(examples),
            'n_parameters': examples.shape[1]
        }

        # Basic statistics
        analysis['mean_params'] = np.mean(examples, axis=0)
        analysis['std_params'] = np.std(examples, axis=0)
        analysis['median_params'] = np.median(examples, axis=0)

        # Overall variance (higher = less consistent)
        analysis['total_variance'] = np.mean(np.var(examples, axis=0))
        analysis['consistency_score'] = 1.0 / (1.0 + analysis['total_variance'])

        # Coefficient of variation (normalized variance)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = analysis['std_params'] / np.abs(analysis['mean_params'])
            cv[~np.isfinite(cv)] = 0
        analysis['coefficient_of_variation'] = np.mean(cv)

        # Cluster analysis (if enough examples)
        if len(examples) >= 10:
            cluster_metrics = self._analyze_clusters(examples)
            analysis.update(cluster_metrics)

        # Per-band analysis (assuming 5-band EQ: 15 params = 5 bands * 3 params)
        if examples.shape[1] >= 15:
            analysis['band_analysis'] = self._analyze_bands(examples)

        return analysis

    def _analyze_clusters(self, examples: np.ndarray) -> Dict:
        """
        Perform clustering to see if there are sub-groups within a term
        """

        n_clusters_range = range(2, min(6, len(examples) // 3))

        best_k = 2
        best_score = -1

        for k in n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(examples)
            score = silhouette_score(examples, labels)

            if score > best_score:
                best_score = score
                best_k = k

        # Final clustering with best k
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(examples)

        return {
            'optimal_clusters': best_k,
            'silhouette_score': best_score,
            'davies_bouldin_score': davies_bouldin_score(examples, labels),
            'cluster_sizes': [np.sum(labels == i) for i in range(best_k)],
            'cluster_centers': kmeans.cluster_centers_
        }

    def _analyze_bands(self, examples: np.ndarray) -> List[Dict]:
        """
        Analyze each frequency band separately
        Assumes format: [band1_gain, band1_freq, band1_q, band2_gain, ...]
        """

        n_bands = examples.shape[1] // 3
        band_analysis = []

        for band_idx in range(n_bands):
            gain_idx = band_idx * 3
            freq_idx = band_idx * 3 + 1
            q_idx = band_idx * 3 + 2

            gains = examples[:, gain_idx]

            band_info = {
                'band_index': band_idx,
                'band_name': self.band_names[band_idx] if band_idx < len(self.band_names) else f'Band {band_idx}',
                'mean_gain': np.mean(gains),
                'std_gain': np.std(gains),
                'median_gain': np.median(gains),
                'gain_range': (np.min(gains), np.max(gains)),
                'boost_percentage': np.sum(gains > 0.5) / len(gains) * 100,
                'cut_percentage': np.sum(gains < -0.5) / len(gains) * 100,
                'neutral_percentage': np.sum(np.abs(gains) <= 0.5) / len(gains) * 100
            }

            band_analysis.append(band_info)

        return band_analysis

    def compare_terms(self, terms: List[str]) -> pd.DataFrame:
        """
        Compare multiple semantic terms to see how they differ
        """

        comparison_data = []

        for term in terms:
            analysis = self.analyze_term_consistency(term)

            if 'error' not in analysis:
                comparison_data.append({
                    'Term': term,
                    'Examples': analysis['n_examples'],
                    'Consistency': f"{analysis['consistency_score']:.3f}",
                    'Variance': f"{analysis['total_variance']:.3f}",
                    'Coeff. Variation': f"{analysis['coefficient_of_variation']:.3f}",
                    'Clusters': analysis.get('optimal_clusters', 'N/A'),
                    'Cluster Quality': f"{analysis.get('silhouette_score', 0):.3f}"
                })

        return pd.DataFrame(comparison_data)

    def visualize_term_distribution(self, term: str, output_dir: Path):
        """
        Create comprehensive visualization for a semantic term
        """

        analysis = self.analyze_term_consistency(term)
        examples = self.get_term_examples(term)

        if len(examples) == 0:
            print(f"No examples found for '{term}'")
            return

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Parameter distribution (PCA)
        ax1 = fig.add_subplot(gs[0, :2])
        if examples.shape[1] >= 2:
            pca = PCA(n_components=2)
            examples_2d = pca.fit_transform(examples)

            ax1.scatter(examples_2d[:, 0], examples_2d[:, 1], alpha=0.6, s=50)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax1.set_title(f'"{term}": Parameter Distribution (PCA)\nn={len(examples)} examples')
            ax1.grid(True, alpha=0.3)

        # 2. Consistency metrics
        ax2 = fig.add_subplot(gs[0, 2])
        metrics = {
            'Consistency': analysis['consistency_score'],
            'Cluster\nQuality': analysis.get('silhouette_score', 0)
        }
        bars = ax2.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db'])
        ax2.set_ylim([0, 1])
        ax2.set_title('Quality Metrics')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

        # 3. Per-band gain distribution
        if 'band_analysis' in analysis:
            ax3 = fig.add_subplot(gs[1, :])
            band_data = analysis['band_analysis']

            means = [b['mean_gain'] for b in band_data]
            stds = [b['std_gain'] for b in band_data]
            labels = [b['band_name'] for b in band_data]

            x = np.arange(len(labels))
            bars = ax3.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                          color=['#e74c3c' if m < 0 else '#2ecc71' for m in means])

            ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.set_ylabel('Gain (dB)')
            ax3.set_title(f'"{term}": Mean EQ Curve (±1 std dev)')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add boost/cut/neutral percentages
            for i, b in enumerate(band_data):
                if b['mean_gain'] > 0.5:
                    color_text = '#27ae60'
                    label_text = f"↑{b['boost_percentage']:.0f}%"
                elif b['mean_gain'] < -0.5:
                    color_text = '#c0392b'
                    label_text = f"↓{b['cut_percentage']:.0f}%"
                else:
                    color_text = '#7f8c8d'
                    label_text = f"~{b['neutral_percentage']:.0f}%"

                ax3.text(i, means[i] + stds[i] + 0.3, label_text,
                        ha='center', fontsize=9, color=color_text, weight='bold')

        # 4. Boost/Cut/Neutral breakdown per band
        ax4 = fig.add_subplot(gs[2, 0])
        if 'band_analysis' in analysis:
            boost_pcts = [b['boost_percentage'] for b in band_data]
            cut_pcts = [b['cut_percentage'] for b in band_data]
            neutral_pcts = [b['neutral_percentage'] for b in band_data]

            width = 0.25
            x = np.arange(len(labels))

            ax4.bar(x - width, boost_pcts, width, label='Boost (>0.5dB)', color='#2ecc71')
            ax4.bar(x, neutral_pcts, width, label='Neutral (±0.5dB)', color='#95a5a6')
            ax4.bar(x + width, cut_pcts, width, label='Cut (<-0.5dB)', color='#e74c3c')

            ax4.set_xticks(x)
            ax4.set_xticklabels(labels, rotation=45, ha='right')
            ax4.set_ylabel('Percentage of Examples (%)')
            ax4.set_title('Engineer Behavior per Band')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. Variance heatmap
        ax5 = fig.add_subplot(gs[2, 1])
        if examples.shape[1] >= 15:
            n_bands = examples.shape[1] // 3
            variance_matrix = np.zeros((n_bands, 3))

            for band_idx in range(n_bands):
                variance_matrix[band_idx, 0] = np.var(examples[:, band_idx * 3])      # gain
                variance_matrix[band_idx, 1] = np.var(examples[:, band_idx * 3 + 1])  # freq
                variance_matrix[band_idx, 2] = np.var(examples[:, band_idx * 3 + 2])  # Q

            im = ax5.imshow(variance_matrix.T, cmap='YlOrRd', aspect='auto')
            ax5.set_yticks([0, 1, 2])
            ax5.set_yticklabels(['Gain', 'Freq', 'Q'])
            ax5.set_xticks(range(n_bands))
            ax5.set_xticklabels([b['band_name'] for b in band_data], rotation=45, ha='right')
            ax5.set_title('Parameter Variance by Band')
            plt.colorbar(im, ax=ax5, label='Variance')

        # 6. Statistics summary
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')

        stats_text = f"""
TERM: "{term}"

Examples: {analysis['n_examples']}
Parameters: {analysis['n_parameters']}

Consistency: {analysis['consistency_score']:.3f}
Total Variance: {analysis['total_variance']:.3f}
Coeff. Variation: {analysis['coefficient_of_variation']:.3f}

Clustering:
  Optimal k: {analysis.get('optimal_clusters', 'N/A')}
  Silhouette: {analysis.get('silhouette_score', 'N/A')}
  Davies-Bouldin: {analysis.get('davies_bouldin_score', 'N/A')}

Interpretation:
  Consistency > 0.7: High
  Silhouette > 0.5: Good clusters
  Variance < 1.0: Low variance
        """.strip()

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(f'Comprehensive Analysis: "{term}"', fontsize=16, weight='bold')

        output_path = output_dir / f'semantic_analysis_{term}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()

    def generate_comparison_heatmap(self, terms: List[str], output_dir: Path):
        """
        Create heatmap showing how different terms affect each frequency band
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Collect mean gains for each term
        term_gains = []
        valid_terms = []

        for term in terms:
            examples = self.get_term_examples(term)
            if len(examples) > 0 and examples.shape[1] >= 15:
                n_bands = min(6, examples.shape[1] // 3)
                gains = [np.mean(examples[:, i * 3]) for i in range(n_bands)]
                term_gains.append(gains)
                valid_terms.append(term)

        if not term_gains:
            print("No valid terms for heatmap")
            return

        # Create heatmap
        term_gains_array = np.array(term_gains)

        # Mean gain heatmap
        im1 = ax1.imshow(term_gains_array, cmap='RdBu_r', aspect='auto',
                        vmin=-3, vmax=3, interpolation='nearest')
        ax1.set_yticks(range(len(valid_terms)))
        ax1.set_yticklabels(valid_terms)
        ax1.set_xticks(range(len(self.band_names)))
        ax1.set_xticklabels(self.band_names, rotation=45, ha='right')
        ax1.set_title('Mean Gain per Band (dB)', fontsize=14, weight='bold')
        plt.colorbar(im1, ax=ax1, label='Gain (dB)')

        # Add values as text
        for i in range(len(valid_terms)):
            for j in range(term_gains_array.shape[1]):
                text = ax1.text(j, i, f'{term_gains_array[i, j]:.1f}',
                              ha='center', va='center',
                              color='white' if abs(term_gains_array[i, j]) > 1.5 else 'black',
                              fontsize=9)

        # Standard deviation heatmap
        term_stds = []
        for term in valid_terms:
            examples = self.get_term_examples(term)
            n_bands = min(6, examples.shape[1] // 3)
            stds = [np.std(examples[:, i * 3]) for i in range(n_bands)]
            term_stds.append(stds)

        term_stds_array = np.array(term_stds)
        im2 = ax2.imshow(term_stds_array, cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=3, interpolation='nearest')
        ax2.set_yticks(range(len(valid_terms)))
        ax2.set_yticklabels(valid_terms)
        ax2.set_xticks(range(len(self.band_names)))
        ax2.set_xticklabels(self.band_names, rotation=45, ha='right')
        ax2.set_title('Consistency (Std Dev)', fontsize=14, weight='bold')
        plt.colorbar(im2, ax=ax2, label='Std Dev (dB)')

        for i in range(len(valid_terms)):
            for j in range(term_stds_array.shape[1]):
                text = ax2.text(j, i, f'{term_stds_array[i, j]:.1f}',
                              ha='center', va='center',
                              color='white' if term_stds_array[i, j] > 1.5 else 'black',
                              fontsize=9)

        plt.suptitle('Semantic Terms: EQ Characteristics Comparison',
                    fontsize=16, weight='bold')

        output_path = output_dir / 'semantic_comparison_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {output_path}")
        plt.close()

    def generate_report(self, terms: List[str], output_dir: Path):
        """
        Generate markdown report with key findings
        """

        report_lines = [
            "# Semantic Term Analysis Report\n",
            "## What do engineers REALLY mean by semantic terms?\n",
            f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**Dataset**: SocialFX-original\n",
            f"**Terms Analyzed**: {', '.join(terms)}\n",
            "---\n\n"
        ]

        # Comparison table
        report_lines.append("## 1. Term Comparison Summary\n\n")
        comparison_df = self.compare_terms(terms)
        report_lines.append(comparison_df.to_markdown(index=False))
        report_lines.append("\n\n")

        # Individual term insights
        report_lines.append("## 2. Detailed Term Analysis\n\n")

        for term in terms:
            analysis = self.analyze_term_consistency(term)

            if 'error' in analysis:
                continue

            report_lines.append(f"### {term.upper()}\n\n")
            report_lines.append(f"- **Examples**: {analysis['n_examples']}\n")
            report_lines.append(f"- **Consistency Score**: {analysis['consistency_score']:.3f} ")

            if analysis['consistency_score'] > 0.7:
                report_lines.append("(High - engineers agree)\n")
            elif analysis['consistency_score'] > 0.5:
                report_lines.append("(Medium - some variation)\n")
            else:
                report_lines.append("(Low - high variation)\n")

            report_lines.append(f"- **Variance**: {analysis['total_variance']:.3f}\n")

            if 'optimal_clusters' in analysis:
                report_lines.append(f"- **Natural Clusters**: {analysis['optimal_clusters']} ")
                report_lines.append(f"(suggests {analysis['optimal_clusters']} distinct interpretations)\n")

            # Band-specific insights
            if 'band_analysis' in analysis:
                report_lines.append("\n**EQ Character**:\n\n")

                for band in analysis['band_analysis']:
                    if abs(band['mean_gain']) > 0.5:
                        direction = "boost" if band['mean_gain'] > 0 else "cut"
                        report_lines.append(f"- {band['band_name']}: {band['mean_gain']:+.1f} dB ({direction}) ")
                        report_lines.append(f"[{band['boost_percentage']:.0f}% boost, {band['cut_percentage']:.0f}% cut]\n")

            report_lines.append("\n")

        # Key findings
        report_lines.append("## 3. Key Findings\n\n")
        report_lines.append("### Consistency Rankings\n\n")

        analyses = [(term, self.analyze_term_consistency(term)) for term in terms]
        analyses = [(t, a) for t, a in analyses if 'error' not in a]
        analyses.sort(key=lambda x: x[1]['consistency_score'], reverse=True)

        report_lines.append("Most consistent terms (engineers agree):\n\n")
        for i, (term, analysis) in enumerate(analyses[:5], 1):
            report_lines.append(f"{i}. **{term}**: {analysis['consistency_score']:.3f}\n")

        report_lines.append("\nMost varied terms (different interpretations):\n\n")
        for i, (term, analysis) in enumerate(reversed(analyses[-5:]), 1):
            report_lines.append(f"{i}. **{term}**: {analysis['consistency_score']:.3f}\n")

        report_lines.append("\n---\n\n")
        report_lines.append("*Generated by semantic_term_analysis.py*\n")

        # Write report
        report_path = output_dir / 'semantic_analysis_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)

        print(f"[OK] Saved: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Term Analysis")
    parser.add_argument('--terms', nargs='+',
                       default=['warm', 'bright', 'punchy', 'smooth', 'heavy', 'soft', 'harsh'],
                       help='Semantic terms to analyze')
    parser.add_argument('--output', default='./semantic_analysis_results',
                       help='Output directory')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only generate comparison, skip individual visualizations')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("SEMANTIC TERM ANALYSIS")
    print("=" * 60)
    print(f"Analyzing {len(args.terms)} terms: {', '.join(args.terms)}")
    print(f"Output directory: {output_dir}")
    print()

    # Initialize analyzer
    analyzer = SemanticTermAnalyzer()

    # Generate comparison table
    print("Generating comparison table...")
    comparison_df = analyzer.compare_terms(args.terms)
    print("\n" + "=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60 + "\n")

    # Generate visualizations
    if not args.compare_only:
        print("Generating individual term visualizations...")
        for term in args.terms:
            print(f"  Analyzing '{term}'...")
            analyzer.visualize_term_distribution(term, output_dir)

    # Generate comparison heatmap
    print("\nGenerating comparison heatmap...")
    analyzer.generate_comparison_heatmap(args.terms, output_dir)

    # Generate report
    print("Generating analysis report...")
    analyzer.generate_report(args.terms, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - semantic_analysis_[term].png  (one per term)")
    print("  - semantic_comparison_heatmap.png")
    print("  - semantic_analysis_report.md")
    print("\nThese visualizations are perfect for your report!")


if __name__ == '__main__':
    main()
