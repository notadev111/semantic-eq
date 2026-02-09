"""
Select 12 tracks with extreme/distinct audio features from FMA small subset
for use in a listening test.
"""
import pandas as pd
import numpy as np

BASE = 'c:/Users/danie/Documents/!ELEC0030 Project/semantic_mastering_system'

# -- 1. Read CSVs -------------------------------------------------------------
print("=" * 80)
print("STEP 1: Loading data")
print("=" * 80)

tracks = pd.read_csv(
    f'{BASE}/data/fma/fma_metadata/fma_metadata/tracks.csv',
    header=[0, 1], index_col=0
)
echonest = pd.read_csv(
    f'{BASE}/data/fma/fma_metadata/fma_metadata/echonest.csv',
    header=[0, 1, 2], index_col=0
)

print(f"  tracks.csv  : {tracks.shape[0]} tracks, {tracks.shape[1]} columns")
print(f"  echonest.csv: {echonest.shape[0]} tracks, {echonest.shape[1]} columns")

# -- 2. Filter to 'small' subset ----------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: Filtering to FMA small subset")
print("=" * 80)

small_mask = tracks[('set', 'subset')] == 'small'
tracks_small = tracks[small_mask]
print(f"  Tracks in 'small' subset: {len(tracks_small)}")

# -- 3. Extract echonest audio features ---------------------------------------
print("\n" + "=" * 80)
print("STEP 3: Extracting echonest audio features")
print("=" * 80)

feature_names = [
    'energy', 'acousticness', 'danceability',
    'instrumentalness', 'speechiness', 'valence', 'tempo'
]
feat_cols = [('echonest', 'audio_features', f) for f in feature_names]
audio_feat = echonest[feat_cols].copy()
audio_feat.columns = feature_names          # flatten to simple names
audio_feat = audio_feat.apply(pd.to_numeric, errors='coerce')
print(f"  Echonest tracks with audio features: {len(audio_feat)}")
print(f"  Features: {feature_names}")

# -- 4. Inner join on track_id -------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4: Inner join - small subset AND echonest features")
print("=" * 80)

common_ids = tracks_small.index.intersection(audio_feat.index)
df = audio_feat.loc[common_ids].copy()

# Add genre
df['genre'] = tracks.loc[common_ids, ('track', 'genre_top')]
df = df.dropna(subset=feature_names)        # drop rows with NaN features
print(f"  Tracks with both small-subset membership and echonest features: {len(df)}")
print(f"  Genre distribution:\n{df['genre'].value_counts().to_string()}")

# -- 5-6. Select 12 tracks with extreme features ------------------------------
print("\n" + "=" * 80)
print("STEP 5-6: Selecting 12 tracks with EXTREME features")
print("=" * 80)

selected = {}  # track_id -> info dict
used_ids = set()


def pick(pool, n, category, reason_fn, prefer_diverse_genre=True):
    """Pick n tracks from pool, avoiding duplicates, preferring genre diversity."""
    picks = []
    used_genres = set()
    for tid in pool.index:
        if tid in used_ids:
            continue
        genre = df.loc[tid, 'genre']
        if prefer_diverse_genre and genre in used_genres and len(pool) > n * 3:
            continue
        picks.append(tid)
        used_ids.add(tid)
        used_genres.add(genre)
        feats = {f: float(df.loc[tid, f]) for f in feature_names}
        selected[tid] = {
            'category': category,
            'genre': genre,
            'features': feats,
            'reason': reason_fn(tid)
        }
        if len(picks) == n:
            break
    # If genre-diversity was too strict, relax and fill remaining
    if len(picks) < n:
        for tid in pool.index:
            if tid in used_ids:
                continue
            picks.append(tid)
            used_ids.add(tid)
            genre = df.loc[tid, 'genre']
            feats = {f: float(df.loc[tid, f]) for f in feature_names}
            selected[tid] = {
                'category': category,
                'genre': genre,
                'features': feats,
                'reason': reason_fn(tid)
            }
            if len(picks) == n:
                break
    return picks


# Category A: 2 very high energy
pool_a = df.sort_values('energy', ascending=False)
pick(pool_a, 2, 'A: Very High Energy',
     lambda t: f"energy={df.loc[t,'energy']:.4f} (top of dataset)")

# Category B: 2 very low energy + high acousticness (quiet/gentle)
df['_score_b'] = (1 - df['energy']) + df['acousticness']
pool_b = df.sort_values('_score_b', ascending=False)
pick(pool_b, 2, 'B: Low Energy + High Acousticness (quiet/gentle)',
     lambda t: f"energy={df.loc[t,'energy']:.4f}, acousticness={df.loc[t,'acousticness']:.4f}")

# Category C: 2 low acousticness + high energy (bright/electronic)
df['_score_c'] = df['energy'] + (1 - df['acousticness'])
pool_c = df.sort_values('_score_c', ascending=False)
pick(pool_c, 2, 'C: Low Acousticness + High Energy (bright/electronic)',
     lambda t: f"energy={df.loc[t,'energy']:.4f}, acousticness={df.loc[t,'acousticness']:.4f}")

# Category D: 2 high speechiness
pool_d = df.sort_values('speechiness', ascending=False)
pick(pool_d, 2, 'D: High Speechiness (vocal/speech-heavy)',
     lambda t: f"speechiness={df.loc[t,'speechiness']:.4f}")

# Category E: 2 high instrumentalness
pool_e = df.sort_values('instrumentalness', ascending=False)
pick(pool_e, 2, 'E: High Instrumentalness (no vocals)',
     lambda t: f"instrumentalness={df.loc[t,'instrumentalness']:.4f}")

# Category F: 2 high danceability
pool_f = df.sort_values('danceability', ascending=False)
pick(pool_f, 2, 'F: High Danceability',
     lambda t: f"danceability={df.loc[t,'danceability']:.4f}")

# Clean up temp columns
df.drop(columns=['_score_b', '_score_c'], inplace=True)


# -- 7. Print results ---------------------------------------------------------
def track_id_to_path(tid):
    """Convert track_id (int) to FMA zero-padded path: e.g. 10 -> 000/000010.mp3"""
    tid_str = f"{int(tid):06d}"
    folder = tid_str[:3]
    return f"{folder}/{tid_str}.mp3"


print("\n" + "=" * 80)
print("SELECTED 12 TRACKS FOR LISTENING TEST")
print("=" * 80)

for i, (tid, info) in enumerate(selected.items(), 1):
    path = track_id_to_path(tid)
    print(f"\n  [{i:2d}] Track ID: {int(tid)}")
    print(f"       File:     fma_small/{path}")
    print(f"       Genre:    {info['genre']}")
    print(f"       Category: {info['category']}")
    print(f"       Reason:   {info['reason']}")
    feat = info['features']
    print(f"       Features: energy={feat['energy']:.4f}  acousticness={feat['acousticness']:.4f}  "
          f"danceability={feat['danceability']:.4f}  instrumentalness={feat['instrumentalness']:.4f}  "
          f"speechiness={feat['speechiness']:.4f}  valence={feat['valence']:.4f}  tempo={feat['tempo']:.1f}")

# -- 8. Print tar command -----------------------------------------------------
print("\n" + "=" * 80)
print("TAR COMMAND TO BUNDLE ALL 12 FILES")
print("=" * 80)

fma_small_dir = f"{BASE}/data/fma/fma_small"
paths = [f"fma_small/{track_id_to_path(tid)}" for tid in selected]
tar_cmd = f"tar -cf listening_test_tracks.tar -C \"{BASE}/data/fma\" \\n  " + " \\n  ".join(paths)
print(f"\n{tar_cmd}\n")

# Also print a simple list of track IDs
print("Track IDs (for reference):", [int(t) for t in selected.keys()])
