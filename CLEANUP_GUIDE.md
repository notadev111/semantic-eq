# Cleanup Guide - Redundant Files

This guide identifies old, redundant, and superseded files in the repository.

## Current Active System (Keep These)

### Core Models
- `neural_eq_safedb_v2.pt` - Pre-trained EQ Encoder/Decoder (V2 system)
- `audio_encoder_best.pt` - Trained Audio Encoder (just completed 100 epochs)

### Core Modules (Keep)
- `core/neural_eq_morphing_safedb_v2.py` - V2 EQ system (CURRENT)
- `core/audio_encoder.py` - Audio Encoder architecture
- `core/adaptive_eq_generator.py` - Adaptive EQ generation (uses both models)
- `core/training_data_synthesis.py` - Synthesizes training data for Audio Encoder
- `core/safe_db_loader_v2.py` - Dataset loader with log-scale normalization

### Training Scripts (Keep)
- `train_neural_eq_safedb_v2.py` - Trains EQ Encoder/Decoder V2
- `train_audio_encoder.py` - Trains Audio Encoder (just used this)

### Testing Scripts (Keep)
- `test_with_real_audio.py` - Test adaptive EQ with real audio files
- `diagnose_audio_encoder.py` - Diagnostic tests for Audio Encoder
- `check_hardware.py` - Check GPU/CPU capabilities
- `quick_sanity_check.py` - Quick test that everything works

### Demo Scripts (Keep)
- `demo_adaptive_eq.py` - Interactive demo of adaptive EQ system

### Documentation (Keep)
- `README.md` - Main documentation
- `CLUSTER_SETUP.md` - Cluster training guide (just created)
- `READY_FOR_CLUSTER.md` - Quick start for cluster (just created)
- `TRAINING_LOG_2026-01-14.md` - Training session log (just created)
- `requirements_cluster.txt` - Cluster dependencies (just created)

---

## Files to DELETE (Redundant/Old)

### Old Model Versions (Superseded by V2)

**DELETE**: `train_neural_eq.py`
- Reason: Original version without SAFE-DB dataset, superseded by `train_neural_eq_safedb_v2.py`

**DELETE**: `train_neural_eq_FILTERED.py`
- Reason: Experimental version with filtering, not used

**DELETE**: `train_neural_eq_safedb.py`
- Reason: V1 of SAFE-DB training, superseded by V2 (log-scale normalization)

**DELETE**: `train_neural_eq_safedb_v3.py`
- Reason: Experimental V3, never completed/used

**DELETE**: `core/neural_eq_morphing.py`
- Reason: Original version without SAFE-DB, superseded by V2

**DELETE**: `core/neural_eq_morphing_safedb.py`
- Reason: V1 of SAFE-DB system, superseded by V2

**DELETE**: `core/neural_eq_morphing_lite.py`
- Reason: Simplified version, not used in current system

### Old Test Scripts (Superseded)

**DELETE**: `test_trained_model.py`
- Reason: Old test script, superseded by `test_with_real_audio.py` and `diagnose_audio_encoder.py`

**DELETE**: `test_safedb_model.py`
- Reason: V1 test script, superseded by newer tests

**DELETE**: `test_safedb_model_v2.py`
- Reason: Old V2 test script, superseded by `test_with_real_audio.py`

**DELETE**: `test_safedb_model_v3.py`
- Reason: V3 never completed, not used

**DELETE**: `test_adaptive_system.py`
- Reason: Old adaptive system test, superseded by `demo_adaptive_eq.py` and `test_with_real_audio.py`

**DELETE**: `quick_start.py`
- Reason: Old quick start for original system, superseded by current demos

### Old Analysis/Utility Scripts

**DELETE**: `analyze_clustering_failure.py`
- Reason: One-time analysis script for debugging, no longer needed

**DELETE**: `analyze_safe_dataset.py`
- Reason: One-time analysis, dataset already analyzed

**DELETE**: `analyze_training_results.py`
- Reason: Old analysis script, W&B now provides this

**DELETE**: `count_words.py`
- Reason: Utility script, not part of core system

**DELETE**: `decode_safe_eq_params.py`
- Reason: One-time utility for decoding EQ params, functionality now in core modules

**DELETE**: `check_training_history.py`
- Reason: Old training history checker, W&B provides this now

**DELETE**: `visualize_latent_space.py`
- Reason: One-time visualization script, W&B provides this now

### Old Application Scripts

**DELETE**: `apply_neural_eq_v2.py`
- Reason: Old standalone application, superseded by `demo_adaptive_eq.py`

**DELETE**: `apply_neural_eq_v2_simple.py`
- Reason: Simplified version, not used

### Research/Experimental Files

**MOVE TO `research/` or DELETE**:
- All files in `research/` folder are experimental and can be archived or deleted if not actively used

---

## Recommended Cleanup Actions

### Option 1: Archive (Safest)
Create an `archive/` folder and move old files there:
```bash
mkdir archive
mv train_neural_eq.py archive/
mv train_neural_eq_FILTERED.py archive/
mv train_neural_eq_safedb.py archive/
# ... etc
```

### Option 2: Delete (Clean)
Delete files directly if you're confident you won't need them:
```bash
rm train_neural_eq.py
rm train_neural_eq_FILTERED.py
rm train_neural_eq_safedb.py
# ... etc
```

### Option 3: Git Branch (Version Control)
Create a branch with old files before deleting:
```bash
git checkout -b archive-old-files
git add .
git commit -m "Archive old files before cleanup"
git checkout main
# Now delete files on main
```

---

## Summary of Current System Architecture

**Two-Model System**:

1. **neural_eq_safedb_v2.pt** (EQ Encoder/Decoder)
   - Input: EQ parameters
   - Output: 32-dim latent + reconstructed EQ
   - Trained on: SAFE-DB dataset
   - Status: Pre-trained, frozen

2. **audio_encoder_best.pt** (Audio Encoder)
   - Input: Audio waveform
   - Output: 32-dim latent (same space as EQ Encoder)
   - Trained on: Synthesized audio
   - Status: Just trained 100 epochs

**How They Work Together**:
```
Audio → [Audio Encoder] → z_audio (32-dim)
                              ↓
                        Latent traversal
                              ↓
"warm" → [EQ Encoder] → z_warm (32-dim)
                              ↓
              z_final = z_audio + intensity × (z_warm - z_audio)
                              ↓
                      [EQ Decoder] → EQ parameters
```

**Active Scripts**:
- Training: `train_neural_eq_safedb_v2.py`, `train_audio_encoder.py`
- Testing: `test_with_real_audio.py`, `diagnose_audio_encoder.py`
- Demo: `demo_adaptive_eq.py`
- Utilities: `check_hardware.py`, `quick_sanity_check.py`

---

## Files Count

**Current active files**: ~15 scripts + core modules
**Redundant/old files**: ~20 scripts
**Potential space savings**: Significant reduction in clutter

After cleanup, the repository will be much cleaner and easier to navigate!
