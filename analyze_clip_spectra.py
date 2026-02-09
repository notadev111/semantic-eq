"""
Analyze spectral characteristics of listening test clips.
Find out if they actually have different tonal profiles.
"""
import numpy as np
from pathlib import Path

def analyze_spectral_profile(audio_path):
    """Compute spectral centroid, rolloff, and bass/treble ratio."""
    import soundfile as sf
    import librosa

    audio, sr = sf.read(str(audio_path))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0].mean()

    # Bass vs treble energy ratio
    S = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = freqs < 300
    mid_mask = (freqs >= 300) & (freqs < 2000)
    treble_mask = freqs >= 2000

    bass_energy = S[bass_mask].mean()
    mid_energy = S[mid_mask].mean()
    treble_energy = S[treble_mask].mean()

    rms = np.sqrt(np.mean(audio**2))

    return {
        "centroid": centroid,
        "rolloff": rolloff,
        "bandwidth": bandwidth,
        "bass_energy": bass_energy,
        "mid_energy": mid_energy,
        "treble_energy": treble_energy,
        "bass_treble_ratio": bass_energy / (treble_energy + 1e-8),
        "rms": rms,
    }


clips = [
    ("001544", "Folk - Quiet acoustic"),
    ("006367", "International - Vocal"),
    ("014542", "Hip-Hop - Rap"),
    ("014570", "Instrumental - Quiet"),
    ("028802", "Pop - High energy"),
    ("030486", "Pop - Danceable"),
    ("040242", "Folk - Instrumental"),
    ("048293", "Electronic - Instrumental"),
    ("071508", "Electronic - Bright"),
    ("089846", "Hip-Hop - Danceable"),
    ("115761", "Rock - Max energy"),
    ("115765", "Rock - Loud"),
]

base = Path("listening_test_tracks")

print(f"{'Clip':<8} {'Description':<28} {'Centroid':>8} {'Rolloff':>8} {'B/T Ratio':>10} {'RMS':>6} {'Tonal Character'}")
print("-" * 95)

results = []
for clip_id, desc in clips:
    folder = clip_id[:3]
    path = base / folder / f"{clip_id}.mp3"
    if not path.exists():
        print(f"{clip_id:<8} {desc:<28} FILE NOT FOUND")
        continue

    spec = analyze_spectral_profile(path)
    results.append((clip_id, desc, spec))

    # Determine tonal character
    if spec["bass_treble_ratio"] > 3.0:
        character = "WARM/DARK (bass heavy)"
    elif spec["bass_treble_ratio"] > 1.5:
        character = "BALANCED-WARM"
    elif spec["bass_treble_ratio"] > 0.8:
        character = "BALANCED"
    elif spec["bass_treble_ratio"] > 0.4:
        character = "BALANCED-BRIGHT"
    else:
        character = "BRIGHT (treble heavy)"

    print(f"{clip_id:<8} {desc:<28} {spec['centroid']:>8.0f} {spec['rolloff']:>8.0f} {spec['bass_treble_ratio']:>10.2f} {spec['rms']:>6.3f} {character}")

# Find the spread
centroids = [r[2]["centroid"] for r in results]
ratios = [r[2]["bass_treble_ratio"] for r in results]
print(f"\nSpectral centroid range: {min(centroids):.0f} - {max(centroids):.0f} Hz")
print(f"Bass/treble ratio range: {min(ratios):.2f} - {max(ratios):.2f}")
print(f"\nIf these are all similar, the clips have similar tonal profiles")
print(f"despite having different genres/energy levels.")
