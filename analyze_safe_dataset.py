"""
Analyze SAFE-DB Dataset Structure
==================================
Understand the structure of SAFEEqualiserUserData and SAFEEqualiserAudioFeatureData
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load datasets
user_data_path = Path("research/data/SAFEEqualiserUserData.csv")
audio_data_path = Path("research/data/SAFEEqualiserAudioFeatureData.csv")

print("="*70)
print("SAFE-DB DATASET ANALYSIS")
print("="*70)

# Load UserData
user_df = pd.read_csv(user_data_path, header=None)
print(f"\n{'='*70}")
print("SAFE EQUALISER USER DATA")
print(f"{'='*70}")
print(f"Shape: {user_df.shape}")
print(f"\nFirst 5 rows:")
print(user_df.head())

print(f"\nColumn Analysis (first 20 columns):")
for i in range(min(20, len(user_df.columns))):
    sample_val = user_df.iloc[0, i]
    dtype = type(sample_val).__name__
    print(f"  Col {i:2d}: {dtype:10s} | Sample: {str(sample_val)[:50]}")

# Analyze semantic terms
semantic_col = 1  # Based on the output, column 1 has semantic terms
print(f"\n\nSemantic Terms Analysis:")
print(f"  Total unique terms: {user_df[semantic_col].nunique()}")
print(f"  Most common terms:")
term_counts = user_df[semantic_col].value_counts().head(20)
for term, count in term_counts.items():
    print(f"    {term:15s}: {count:4d} examples")

# Try to identify EQ parameter columns
print(f"\n\nIdentifying EQ Parameters:")
print("  Looking for numeric columns that might be EQ params...")

# Columns 3-17 seem to be numeric parameters based on the sample
numeric_cols = []
for i in range(3, 18):
    try:
        vals = pd.to_numeric(user_df[i], errors='coerce')
        if vals.notna().sum() > len(user_df) * 0.5:  # >50% valid numbers
            numeric_cols.append(i)
            print(f"    Col {i:2d}: min={vals.min():8.2f}, max={vals.max():8.2f}, mean={vals.mean():8.2f}")
    except:
        pass

print(f"\n  Found {len(numeric_cols)} numeric parameter columns: {numeric_cols}")

# Load AudioFeatureData
print(f"\n\n{'='*70}")
print("SAFE EQUALISER AUDIO FEATURE DATA")
print(f"{'='*70}")

audio_df = pd.read_csv(audio_data_path, header=None)
print(f"Shape: {audio_df.shape}")
print(f"\nFirst 3 rows:")
print(audio_df.head(3))

# Understand processed vs unprocessed
print(f"\n\nProcessed/Unprocessed Analysis:")
print(f"  Column 1 values: {audio_df[1].unique()}")
print(f"  Counts: {audio_df[1].value_counts()}")

print(f"\n\nRelationship between datasets:")
print(f"  UserData rows: {len(user_df)}")
print(f"  AudioData rows: {len(audio_df)}")
print(f"  Ratio: {len(audio_df) / len(user_df):.1f}x")
print(f"  → Each user setting has ~2 audio feature rows (before/after)")

# Try to match by ID
print(f"\n\nMatching datasets by ID:")
print(f"  UserData IDs: col 0, range {user_df[0].min()} to {user_df[0].max()}")
print(f"  AudioData IDs: col 0, range {audio_df[0].min()} to {audio_df[0].max()}")

# Sample matching
sample_id = user_df.iloc[0, 0]
print(f"\n  Example: User ID {sample_id}")
print(f"    Semantic term: {user_df.iloc[0, 1]}")
matching_audio = audio_df[audio_df[0] == sample_id]
print(f"    Matching audio rows: {len(matching_audio)}")
if len(matching_audio) > 0:
    print(f"    Types: {matching_audio[1].tolist()}")

print(f"\n\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print("""
SAFE-DB Dataset Structure:

1. SAFEEqualiserUserData.csv:
   - Each row = one user's EQ setting with semantic label
   - Column 0: ID
   - Column 1: Semantic descriptor (e.g., "warm", "bright")
   - Column 2: IP address
   - Columns 5-17: EQ parameters (13 bands × parameters)
   - Last column: Hash/UUID

2. SAFEEqualiserAudioFeatureData.csv:
   - Each ID appears twice (processed + unprocessed)
   - Column 0: ID (matches UserData)
   - Column 1: "processed" or "unprocessed"
   - Columns 2+: Audio features (MFCCs, spectral features, etc.)

Dataset Size:
   - 1,699 user EQ settings
   - 368 unique semantic terms
   - Most common: warm (457), bright (421)

This is different from SocialFX which has 1,595 examples with 765 terms!
""")

print("\n\n{'='*70}")
print("RECOMMENDATION FOR YOUR PROJECT")
print(f"{'='*70}")
print("""
SAFE-DB appears to have:
  ✓ More examples per term (warm: 457 vs SocialFX's fewer)
  ✓ Cleaner semantic labels (fewer unique terms, more examples each)
  ✓ Audio features (before/after) for evaluation
  ✓ Likely 13-band EQ (based on parameter columns)

Next steps:
  1. Parse the EQ parameter columns properly (figure out structure)
  2. Understand what the 13 bands represent
  3. Build new semantic_mastering_safe.py
  4. Decide if neural model needs modification for different param count
""")
