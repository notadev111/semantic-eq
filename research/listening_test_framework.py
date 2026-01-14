"""
Listening Test Framework
=========================

Simple framework for conducting perceptual validation with real listeners.
Perfect for undergrad project - not too advanced but methodologically sound.

Generates:
1. Test audio files (A/B pairs)
2. Survey questions (Google Forms compatible)
3. Results analysis scripts

Based on ITU-R BS.1116 recommendations (simplified for student project)

Usage:
    # Step 1: Prepare test files
    python listening_test_framework.py --prepare --audio your_mix.wav --terms warm bright punchy

    # Step 2: After collecting responses, analyze results
    python listening_test_framework.py --analyze --responses responses.csv
"""

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import json
import hashlib
import warnings
warnings.filterwarnings("ignore")

# Import semantic mastering
try:
    from core.semantic_mastering import SemanticMasteringEQ
except ImportError:
    import sys
    sys.path.append('..')
    from core.semantic_mastering import SemanticMasteringEQ


class ListeningTestPreparer:
    """
    Prepare listening test materials
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.mastering = SemanticMasteringEQ(sample_rate=sample_rate)
        self.mastering.initialize()

    def prepare_ab_test(self, audio_path: str, terms: List[str],
                       output_dir: Path) -> Dict:
        """
        Prepare A/B test comparing semantic mastering vs original

        Returns metadata for survey creation
        """

        output_dir.mkdir(exist_ok=True, parents=True)

        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)

        # Take 15-second excerpt (standard for listening tests)
        excerpt_samples = 15 * self.sr
        if audio.shape[1] > excerpt_samples:
            # Take from 30% into the track (usually past intro)
            start = int(audio.shape[1] * 0.3)
            audio = audio[:, start:start + excerpt_samples]

        # Normalize for consistent loudness
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8) * 0.7

        test_metadata = {
            'source_file': audio_path,
            'duration_seconds': audio.shape[1] / self.sr,
            'terms_tested': terms,
            'samples': []
        }

        # Create reference (original)
        reference_path = output_dir / 'reference.wav'
        torchaudio.save(str(reference_path), audio, self.sr)
        print(f"✓ Saved reference: {reference_path}")

        # Create processed versions for each term
        for term in terms:
            print(f"\nProcessing '{term}'...")

            try:
                processed = self.mastering.apply_mastering(audio, term)

                # Loudness match (critical for fair comparison)
                processed = self._loudness_match(processed, audio)

                # Generate random ID for blind testing
                sample_id = hashlib.md5(f"{term}_{audio_path}".encode()).hexdigest()[:8]

                # Save with randomized filename
                output_path = output_dir / f'sample_{sample_id}.wav'
                torchaudio.save(str(output_path), processed, self.sr)

                test_metadata['samples'].append({
                    'id': sample_id,
                    'term': term,
                    'filename': output_path.name,
                    'type': 'processed'
                })

                print(f"  ✓ Saved: {output_path}")

            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # Save metadata
        metadata_path = output_dir / 'test_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(test_metadata, f, indent=2)

        print(f"\n✓ Saved metadata: {metadata_path}")

        return test_metadata

    def _loudness_match(self, audio: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Match loudness to reference (simple RMS matching)
        """
        ref_rms = torch.sqrt(torch.mean(reference ** 2))
        audio_rms = torch.sqrt(torch.mean(audio ** 2))

        if audio_rms > 1e-8:
            return audio * (ref_rms / audio_rms)
        return audio

    def generate_survey_template(self, metadata: Dict, output_dir: Path):
        """
        Generate survey questions for Google Forms or similar
        """

        lines = [
            "# Listening Test Survey Template\n",
            "## Copy these questions into Google Forms or similar\n\n",
            "---\n\n",
            "## Instructions for Participants\n\n",
            "You will listen to pairs of audio samples:\n",
            "- **Reference**: The original audio\n",
            "- **Sample X**: A processed version\n\n",
            "For each pair, rate how well the semantic term describes the change.\n\n",
            "**Important**:\n",
            "- Use good headphones or studio monitors\n",
            "- Listen in a quiet environment\n",
            "- You can replay samples as many times as needed\n",
            "- There are no wrong answers - trust your ears!\n\n",
            "---\n\n"
        ]

        # Generate questions for each sample
        for i, sample in enumerate(metadata['samples'], 1):
            term = sample['term']
            sample_id = sample['id']

            lines.append(f"## Question {i}: Sample {sample_id}\n\n")
            lines.append(f"Listen to:\n")
            lines.append(f"- `reference.wav` (original)\n")
            lines.append(f"- `{sample['filename']}` (processed)\n\n")

            lines.append(f"### {i}.1 Does the processed version sound more \"{term}\" than the original?\n\n")
            lines.append("- [ ] Much less {}\n".format(term))
            lines.append("- [ ] Slightly less {}\n".format(term))
            lines.append("- [ ] About the same\n")
            lines.append("- [ ] Slightly more {}\n".format(term))
            lines.append("- [ ] Much more {}\n\n".format(term))

            lines.append(f"### {i}.2 Overall, do you prefer:\n\n")
            lines.append("- [ ] Original (reference)\n")
            lines.append("- [ ] Processed version\n")
            lines.append("- [ ] No preference\n\n")

            lines.append(f"### {i}.3 Would you use this '{term}' processing on your own music?\n\n")
            lines.append("- [ ] Definitely yes\n")
            lines.append("- [ ] Probably yes\n")
            lines.append("- [ ] Not sure\n")
            lines.append("- [ ] Probably not\n")
            lines.append("- [ ] Definitely not\n\n")

            lines.append("---\n\n")

        # Demographics
        lines.append("## Background Questions\n\n")
        lines.append("### Audio Experience\n\n")
        lines.append("- [ ] Professional audio engineer\n")
        lines.append("- [ ] Amateur producer/musician\n")
        lines.append("- [ ] Music enthusiast\n")
        lines.append("- [ ] Casual listener\n\n")

        lines.append("### Years of audio production experience\n\n")
        lines.append("- [ ] 0-1 years\n")
        lines.append("- [ ] 1-3 years\n")
        lines.append("- [ ] 3-5 years\n")
        lines.append("- [ ] 5+ years\n\n")

        # Save survey template
        survey_path = output_dir / 'survey_template.md'
        with open(survey_path, 'w') as f:
            f.writelines(lines)

        print(f"✓ Saved survey template: {survey_path}")

        # Also generate Google Forms CSV import
        self._generate_google_forms_csv(metadata, output_dir)

    def _generate_google_forms_csv(self, metadata: Dict, output_dir: Path):
        """
        Generate CSV that can be imported to Google Forms
        """

        # Google Forms CSV format is simple: one question per row
        questions = []

        for i, sample in enumerate(metadata['samples'], 1):
            term = sample['term']

            # Question 1: Semantic appropriateness
            questions.append({
                'Question': f'Q{i}.1: Does sample {sample["id"]} sound more "{term}" than the reference?',
                'Type': 'Multiple choice',
                'Options': f'Much less {term};Slightly less {term};About the same;Slightly more {term};Much more {term}'
            })

            # Question 2: Preference
            questions.append({
                'Question': f'Q{i}.2: For sample {sample["id"]}, do you prefer:',
                'Type': 'Multiple choice',
                'Options': 'Original (reference);Processed version;No preference'
            })

            # Question 3: Usability
            questions.append({
                'Question': f'Q{i}.3: Would you use this \'{term}\' processing?',
                'Type': 'Multiple choice',
                'Options': 'Definitely yes;Probably yes;Not sure;Probably not;Definitely not'
            })

        df = pd.DataFrame(questions)
        csv_path = output_dir / 'google_forms_import.csv'
        df.to_csv(csv_path, index=False)

        print(f"✓ Saved Google Forms import: {csv_path}")


class ListeningTestAnalyzer:
    """
    Analyze listening test results
    """

    def analyze_results(self, responses_csv: str, metadata_json: str,
                       output_dir: Path):
        """
        Analyze survey responses

        Expected CSV format:
        - Columns: participant_id, sample_id, semantic_rating, preference, usability
        """

        # Load data
        df = pd.read_csv(responses_csv)
        with open(metadata_json) as f:
            metadata = json.load(f)

        output_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 60)
        print("LISTENING TEST RESULTS ANALYSIS")
        print("=" * 60)
        print(f"Total responses: {len(df)}")
        print(f"Samples tested: {len(metadata['samples'])}\n")

        # Analysis report
        report_lines = [
            "# Listening Test Results\n\n",
            f"**Total Participants**: {df['participant_id'].nunique()}\n",
            f"**Samples Tested**: {len(metadata['samples'])}\n",
            f"**Source Audio**: {metadata['source_file']}\n\n",
            "---\n\n"
        ]

        # Per-term analysis
        report_lines.append("## Results by Semantic Term\n\n")

        for sample in metadata['samples']:
            term = sample['term']
            sample_id = sample['id']

            sample_responses = df[df['sample_id'] == sample_id]

            if len(sample_responses) == 0:
                continue

            report_lines.append(f"### {term.upper()}\n\n")

            # Semantic appropriateness
            semantic_scores = self._convert_to_scores(
                sample_responses['semantic_rating'],
                ['Much less', 'Slightly less', 'About the same', 'Slightly more', 'Much more']
            )
            mean_semantic = np.mean(semantic_scores)
            std_semantic = np.std(semantic_scores)

            report_lines.append(f"**Semantic Appropriateness**: {mean_semantic:.2f} ± {std_semantic:.2f}\n")
            report_lines.append(f"- Score range: -2 (much less) to +2 (much more)\n")

            if mean_semantic > 0.5:
                report_lines.append(f"- ✅ Listeners perceived increase in '{term}' character\n")
            elif mean_semantic < -0.5:
                report_lines.append(f"- ❌ Listeners perceived decrease in '{term}' character\n")
            else:
                report_lines.append(f"- ⚠️ Ambiguous perception\n")

            # Preference
            preference_counts = sample_responses['preference'].value_counts()
            processed_pref = preference_counts.get('Processed version', 0)
            total = len(sample_responses)
            pref_pct = (processed_pref / total * 100) if total > 0 else 0

            report_lines.append(f"\n**Preference**: {pref_pct:.1f}% preferred processed version\n")

            # Usability
            usability_scores = self._convert_to_scores(
                sample_responses['usability'],
                ['Definitely not', 'Probably not', 'Not sure', 'Probably yes', 'Definitely yes']
            )
            mean_usability = np.mean(usability_scores)

            report_lines.append(f"\n**Usability**: {mean_usability:.2f} / 2.0\n")

            if mean_usability > 0.5:
                report_lines.append(f"- ✅ Listeners would use this processing\n")
            else:
                report_lines.append(f"- ❌ Listeners hesitant to use this\n")

            report_lines.append("\n")

        # Summary statistics
        report_lines.append("## Summary Statistics\n\n")
        report_lines.append("### Best Performing Terms\n\n")

        # Calculate overall scores per term
        term_scores = []
        for sample in metadata['samples']:
            sample_responses = df[df['sample_id'] == sample['id']]
            if len(sample_responses) > 0:
                semantic = np.mean(self._convert_to_scores(
                    sample_responses['semantic_rating'],
                    ['Much less', 'Slightly less', 'About the same', 'Slightly more', 'Much more']
                ))
                term_scores.append((sample['term'], semantic))

        term_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (term, score) in enumerate(term_scores, 1):
            report_lines.append(f"{i}. **{term}**: {score:+.2f}\n")

        report_lines.append("\n---\n\n")
        report_lines.append("*Statistical significance tests can be added if needed*\n")

        # Save report
        report_path = output_dir / 'listening_test_results.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)

        print(f"\n✓ Results saved: {report_path}")

        # Generate visualization
        self._visualize_results(df, metadata, output_dir)

    def _convert_to_scores(self, responses: pd.Series, scale: List[str]) -> np.ndarray:
        """Convert Likert scale responses to numerical scores"""

        score_map = {label: i - len(scale)//2 for i, label in enumerate(scale)}
        return responses.map(score_map).fillna(0).values

    def _visualize_results(self, df: pd.DataFrame, metadata: Dict, output_dir: Path):
        """Create visualization of results"""

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Prepare data
        terms = []
        semantic_scores = []
        preference_pcts = []
        usability_scores = []

        for sample in metadata['samples']:
            sample_responses = df[df['sample_id'] == sample['id']]

            if len(sample_responses) == 0:
                continue

            terms.append(sample['term'])

            # Semantic
            scores = self._convert_to_scores(
                sample_responses['semantic_rating'],
                ['Much less', 'Slightly less', 'About the same', 'Slightly more', 'Much more']
            )
            semantic_scores.append(np.mean(scores))

            # Preference
            pref = (sample_responses['preference'] == 'Processed version').sum()
            preference_pcts.append(pref / len(sample_responses) * 100)

            # Usability
            usability = self._convert_to_scores(
                sample_responses['usability'],
                ['Definitely not', 'Probably not', 'Not sure', 'Probably yes', 'Definitely yes']
            )
            usability_scores.append(np.mean(usability))

        # Plot 1: Semantic appropriateness
        colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in semantic_scores]
        axes[0].barh(terms, semantic_scores, color=colors, alpha=0.7)
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Semantic Appropriateness Score')
        axes[0].set_title('Does it sound like the term?')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Plot 2: Preference
        axes[1].barh(terms, preference_pcts, color='#3498db', alpha=0.7)
        axes[1].axvline(x=50, color='black', linestyle='--', linewidth=0.5, label='50%')
        axes[1].set_xlabel('Preference for Processed (%)')
        axes[1].set_title('Listener Preference')
        axes[1].grid(True, alpha=0.3, axis='x')

        # Plot 3: Usability
        colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in usability_scores]
        axes[2].barh(terms, usability_scores, color=colors, alpha=0.7)
        axes[2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_xlabel('Usability Score')
        axes[2].set_title('Would use this processing?')
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        plot_path = output_dir / 'listening_test_results.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved: {plot_path}")
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Listening Test Framework")

    # Mode selection
    parser.add_argument('--prepare', action='store_true',
                       help='Prepare test files and survey')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze collected responses')

    # Preparation arguments
    parser.add_argument('--audio', help='Input audio file for test preparation')
    parser.add_argument('--terms', nargs='+',
                       default=['warm', 'bright', 'punchy'],
                       help='Semantic terms to test')

    # Analysis arguments
    parser.add_argument('--responses', help='CSV file with survey responses')
    parser.add_argument('--metadata', help='JSON file with test metadata')

    # Common
    parser.add_argument('--output', default='./listening_test',
                       help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.prepare:
        if not args.audio:
            print("Error: --audio required for preparation")
            return

        print("=" * 60)
        print("PREPARING LISTENING TEST")
        print("=" * 60)

        preparer = ListeningTestPreparer()
        metadata = preparer.prepare_ab_test(args.audio, args.terms, output_dir)
        preparer.generate_survey_template(metadata, output_dir)

        print("\n" + "=" * 60)
        print("PREPARATION COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Share audio files with participants")
        print("2. Create survey using survey_template.md or import google_forms_import.csv")
        print("3. Collect responses in CSV format")
        print("4. Run: python listening_test_framework.py --analyze --responses results.csv")

    elif args.analyze:
        if not args.responses or not args.metadata:
            print("Error: --responses and --metadata required for analysis")
            return

        analyzer = ListeningTestAnalyzer()
        analyzer.analyze_results(args.responses, args.metadata, output_dir)

    else:
        print("Error: Use --prepare or --analyze")
        parser.print_help()


if __name__ == '__main__':
    main()
