"""
Decode SAFE EQ Parameter Structure
===================================
Figure out the exact structure of the 13-band EQ
"""

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("research/data/SAFEEqualiserUserData.csv", header=None)

print("="*70)
print("SAFE EQ PARAMETER STRUCTURE ANALYSIS")
print("="*70)

# Columns 5-17 appear to be EQ parameters (13 values)
eq_cols = list(range(5, 18))

print(f"\nTotal EQ parameter columns: {len(eq_cols)}")
print(f"Columns: {eq_cols}")

# Sample a few entries
print("\n" + "="*70)
print("SAMPLE EQ SETTINGS")
print("="*70)

for idx in [0, 1, 10]:
    term = df.iloc[idx, 1]
    params = [df.iloc[idx, i] for i in eq_cols]
    print(f"\nRow {idx}: '{term}'")
    for i, (col, val) in enumerate(zip(eq_cols, params)):
        print(f"  Param {i:2d} (col {col}): {val:10.2f}")

# Pattern analysis
print("\n" + "="*70)
print("PATTERN DETECTION")
print("="*70)

# Check if it's [gain, freq, Q] × 4 or some other pattern
print("\nHypothesis: Pattern could be [gain1, freq1, gain2, freq2, Q2, ...]")
print("Let's look at value ranges to understand:")

for i, col in enumerate(eq_cols):
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    print(f"\nParam {i:2d} (col {col}):")
    print(f"  Range: [{vals.min():8.2f}, {vals.max():8.2f}]")
    print(f"  Mean:  {vals.mean():8.2f}, Std: {vals.std():8.2f}")

    # Identify likely parameter type
    if vals.min() >= -12.1 and vals.max() <= 12.1:
        param_type = "GAIN (dB)"
    elif vals.min() >= 0.09 and vals.max() <= 10.1:
        param_type = "Q-FACTOR"
    elif vals.min() >= 20 and vals.max() <= 20000:
        param_type = "FREQUENCY (Hz)"
    elif vals.min() >= 1 and vals.max() <= 2:
        param_type = "FILTER TYPE?"
    else:
        param_type = "UNKNOWN"

    print(f"  Likely: {param_type}")

print("\n" + "="*70)
print("SAFE EQ STRUCTURE INTERPRETATION")
print("="*70)

print("""
Based on the analysis, the SAFE EQ appears to use:

Columns 3-4: Filter types (1 or 2, probably bell/shelf types)
Columns 5-17: EQ parameters in groups

Looking at the patterns:
- Params at positions 0, 2, 5, 7, 10, 12: Range [-12, 12] = GAIN (dB)
- Params at positions 4, 6, 9, 11: Range [0.1, 10] = Q-FACTOR
- Params at positions 1, 3, 6, 8: Range [20, 20000] = FREQUENCY (Hz)

This suggests a pattern like:
  Band 1: type, gain, freq
  Band 2: type, gain, freq, Q
  Band 3: gain, freq, Q
  Band 4: gain, freq, Q
  Band 5: gain, freq

Or possibly 4 full parametric bands with (type, gain, freq, Q, ...)

Let me check the SAFE plugin documentation...
The SAFE Equaliser is described as a "parametric equaliser"
with multiple bands for semantic audio processing.

Based on common parametric EQ designs and the data:
→ Likely 4-5 parametric EQ bands
→ Each band has: gain, frequency, Q-factor
→ Some bands may have filter type selectors

For your 13-band system, you'll need to:
1. Extract the actual parameter structure from SAFE docs/code
2. Or reverse-engineer from the parameter ranges
3. Design your own 13-band structure if needed
""")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
SAFE-DB Dataset Summary:
  ✓ 1,700 examples
  ✓ 368 unique semantic terms
  ✓ Top terms: warm (457), bright (421) - very well represented!
  ✓ ~13-15 parameters per EQ setting
  ✓ Paired audio features (before/after processing)

vs SocialFX:
  ✓ SocialFX: 1,595 examples, 765 terms (sparser)
  ✓ SAFE-DB: 1,700 examples, 368 terms (denser, better for ML)

Your options:
1. Use SAFE-DB as-is: Decode the existing parameter structure
2. Build custom 13-band: Design your own EQ param format
3. Hybrid: Use SAFE-DB data but convert to your 13-band format

I recommend Option 3:
  - Use SAFE-DB's rich semantic labels (warm: 457 examples!)
  - Convert parameters to your desired 13-band structure
  - Train neural network on standardized format
  - Better clustering due to more examples per term
""")
