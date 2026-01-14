"""
Test SAFE-DB Trained Model
Test the actual EQ generation quality
"""
from core.neural_eq_morphing_safedb import NeuralEQMorphingSAFEDB
import numpy as np

print("="*70)
print("TESTING SAFE-DB TRAINED MODEL")
print("="*70)

# Load model
system = NeuralEQMorphingSAFEDB()
print("\nLoading model from: neural_eq_safedb.pt")
system.load_model("neural_eq_safedb.pt")

# Load dataset (needed for generation)
print("\nLoading dataset (needed for EQ generation)...")
system.load_dataset(min_examples=5, include_audio=False)

# Show available terms
print(f"\nAvailable semantic terms ({len(system.idx_to_term)}):")
for idx, term in system.idx_to_term.items():
    print(f"  {idx+1:2d}. {term}")

# Test EQ generation for key terms
print("\n" + "="*70)
print("EQ PARAMETER GENERATION TEST")
print("="*70)

test_terms = ['warm', 'bright', 'test', 'clear', 'tinny', 'muddy']

for term in test_terms:
    print(f"\n{'='*70}")
    print(f"Term: '{term}'")
    print("="*70)

    try:
        params = system.generate_eq_from_term(term)

        print(f"\nGenerated EQ Parameters:")
        print(f"  Band 1 (Low Shelf):  Gain={params[0]:7.2f}dB, Freq={params[1]:8.1f}Hz")
        print(f"  Band 2 (Bell):       Gain={params[2]:7.2f}dB, Freq={params[3]:8.1f}Hz, Q={params[4]:5.2f}")
        print(f"  Band 3 (Bell):       Gain={params[5]:7.2f}dB, Freq={params[6]:8.1f}Hz, Q={params[7]:5.2f}")
        print(f"  Band 4 (Bell):       Gain={params[8]:7.2f}dB, Freq={params[9]:8.1f}Hz, Q={params[10]:5.2f}")
        print(f"  Band 5 (High Shelf): Gain={params[11]:7.2f}dB, Freq={params[12]:8.1f}Hz")

        # Check if parameters are realistic
        print(f"\nParameter Validity Check:")

        # Check gains
        gains = [params[0], params[2], params[5], params[8], params[11]]
        valid_gains = all(-15 <= g <= 15 for g in gains)
        print(f"  Gains in range [-12, +12] dB: {'YES' if valid_gains else 'NO'}")

        # Check frequencies
        freqs = [params[1], params[3], params[6], params[9], params[12]]
        valid_freqs = all(20 <= f <= 22000 for f in freqs)
        print(f"  Frequencies in range [20, 22000] Hz: {'YES' if valid_freqs else 'NO'}")

        # Check Q values
        qs = [params[4], params[7], params[10]]
        valid_qs = all(0.1 <= q <= 10 for q in qs)
        print(f"  Q values in range [0.1, 10]: {'YES' if valid_qs else 'NO'}")

        # Check ordering (freqs should generally increase)
        ordered = all(freqs[i] <= freqs[i+1] for i in range(len(freqs)-1))
        print(f"  Frequencies ordered low-to-high: {'YES' if ordered else 'NO (overlapping bands)'}")

        if valid_gains and valid_freqs and valid_qs:
            print(f"\n  STATUS: VALID EQ parameters")
        else:
            print(f"\n  STATUS: INVALID - parameters out of range")

    except ValueError as e:
        print(f"\nERROR: {e}")

# Test interpolation
print("\n" + "="*70)
print("SEMANTIC INTERPOLATION TEST: warm -> bright")
print("="*70)

try:
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        params = system.interpolate_terms('warm', 'bright', alpha)
        print(f"\nalpha={alpha:.2f} ({int((1-alpha)*100)}% warm, {int(alpha*100)}% bright):")
        print(f"  Band 1 Gain: {params[0]:6.2f}dB, Band 5 Gain: {params[11]:6.2f}dB")
        print(f"  Band 1 Freq: {params[1]:7.1f}Hz, Band 5 Freq: {params[12]:7.1f}Hz")

    print(f"\nInterpolation should show smooth transition from warm to bright")

except ValueError as e:
    print(f"\nERROR: {e}")

# Evaluate clustering
print("\n" + "="*70)
print("CLUSTERING EVALUATION")
print("="*70)

print("\nEvaluating clustering (dataset already loaded)...")
metrics = system.evaluate_clustering()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nModel has {len(system.idx_to_term)} semantic terms")
print(f"Silhouette score: {metrics['silhouette']:.4f}")
print(f"Davies-Bouldin index: {metrics['davies_bouldin']:.4f}")

if metrics['silhouette'] > 0.5:
    print("\nCLUSTERING: EXCELLENT")
elif metrics['silhouette'] > 0.3:
    print("\nCLUSTERING: GOOD")
elif metrics['silhouette'] > 0.0:
    print("\nCLUSTERING: MODERATE - could be improved")
else:
    print("\nCLUSTERING: POOR - needs retraining")

print("\n" + "="*70)
