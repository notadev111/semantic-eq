import pandas as pd
import numpy as np

BASE = r"c:/Users/danie/Documents/!ELEC0030 Project/semantic_mastering_system"
META = BASE + "/data/fma/fma_metadata/fma_metadata"
NL = chr(10)
SEP = "=" * 90

print("Loading tracks.csv ...")
tracks = pd.read_csv(META + "/tracks.csv", index_col=0, header=[0, 1])
print("  tracks shape:", tracks.shape)
print("Loading echonest.csv ...")
echonest = pd.read_csv(META + "/echonest.csv", index_col=0, header=[0, 1, 2])
print("  echonest shape:", echonest.shape)

small_mask = tracks[("set", "subset")] == "small"
small_ids = tracks.index[small_mask]
print("fma_small tracks:", len(small_ids))
common_ids = small_ids.intersection(echonest.index)
print("Tracks with echonest features:", len(common_ids))

feature_names = ["energy", "acousticness", "danceability", "instrumentalness", "speechiness", "valence", "tempo"]
df = pd.DataFrame(index=common_ids)
df.index.name = "track_id"
for feat in feature_names:
    df[feat] = echonest.loc[common_ids, ("echonest", "audio_features", feat)].astype(float)
df["genre"] = tracks.loc[common_ids, ("track", "genre_top")]
print(f"Final dataset: {len(df)} tracks")
print("Genre distribution:")
print(df["genre"].value_counts())

KEEP_IDS = [1544, 14570, 71508, 115765, 48293, 40242, 89846, 28802]
BAD_IDS = [11782, 32437, 32330, 42235]

def fmt(tid, r):
    fp = str(tid).zfill(6)
    fp = fp[:3] + "/" + fp + ".mp3"
    return (f"  {tid:>6d}  {fp:>14s}  genre={r.genre:<15s}  "
            f"energy={r.energy:.3f}  acoustic={r.acousticness:.3f}  "
            f"dance={r.danceability:.3f}  instr={r.instrumentalness:.3f}  "
            f"speech={r.speechiness:.3f}  valence={r.valence:.3f}  "
            f"tempo={r.tempo:.1f}")

print(NL + SEP)
print("KEEP TRACKS (8 confirmed good):")
print(SEP)
for tid in KEEP_IDS:
    if tid in df.index:
        print(fmt(tid, df.loc[tid]))

print(NL + "BAD TRACKS (to replace):")
for tid in BAD_IDS:
    if tid in df.index:
        print(fmt(tid, df.loc[tid]))

exclude_ids = set(KEEP_IDS + BAD_IDS)
candidates = df[~df.index.isin(exclude_ids)].copy()
candidates = candidates[candidates["speechiness"] <= 0.8]
print(f"{NL}Candidate pool (excl keep/bad, speechiness<=0.8): {len(candidates)}")
keep_genres = set(df.loc[[t for t in KEEP_IDS if t in df.index], "genre"])
print(f"Genres in KEEP set: {keep_genres}")
# REPLACEMENT 1: High energy, NOT Electronic
print(NL + SEP)
print("SEARCH 1: High energy replacement for 32437 - NOT Electronic")
print(SEP)
c1 = candidates[candidates["genre"] != "Electronic"].sort_values("energy", ascending=False)
print("  Top 10 non-Electronic by energy:")
for tid, r in c1.head(10).iterrows():
    print(fmt(tid, r))
rep1 = c1.index[0]
print(f"{NL}  >>> SELECTED: {rep1}")

# REPLACEMENT 2: Danceable, NOT Hip-Hop
print(NL + SEP)
print("SEARCH 2: Danceable replacement for 32330 - NOT Hip-Hop")
print(SEP)
c2 = candidates[(candidates["genre"] != "Hip-Hop") & (~candidates.index.isin([rep1]))]
c2 = c2.sort_values("danceability", ascending=False)
print("  Top 10 non-Hip-Hop by danceability:")
for tid, r in c2.head(10).iterrows():
    print(fmt(tid, r))
rep2 = c2.index[0]
print(f"{NL}  >>> SELECTED: {rep2}")

# REPLACEMENTS 3 and 4: Vocal tracks, speechiness 0.3-0.7
print(NL + SEP)
print("SEARCH 3 and 4: Vocal tracks speechiness 0.3-0.7")
print(SEP)
c34 = candidates[(candidates["speechiness"] >= 0.3) & (candidates["speechiness"] <= 0.7)]
c34 = c34[~c34.index.isin([rep1, rep2])]
c34 = c34[c34["energy"] >= 0.2]
c34 = c34.sort_values("speechiness", ascending=False)
print(f"  Candidates: {len(c34)}")
print(f"  Genre dist: {dict(c34.genre.value_counts())}")
print("  Top 15:")
for tid, r in c34.head(15).iterrows():
    print(fmt(tid, r))

hh = c34[c34["genre"] == "Hip-Hop"]
print(f"{NL}  Hip-Hop vocal candidates: {len(hh)}")
if len(hh) > 0:
    print("  Top 5 Hip-Hop vocal tracks:")
    for tid, r in hh.head(5).iterrows():
        print(fmt(tid, r))
    rep3 = hh.index[0]
else:
    rep3 = c34.index[0]
print(f"  >>> SELECTED for 11782 replacement: {rep3}")

ov = c34[(c34["genre"] != "Hip-Hop") & (~c34.index.isin([rep3]))]
print(f"{NL}  Non-Hip-Hop vocal candidates: {len(ov)}")
iv = ov[ov["genre"] == "International"]
if len(iv) > 0:
    print(f"  International vocal candidates: {len(iv)}")
    print("  Top 5 International vocal tracks:")
    for tid, r in iv.head(5).iterrows():
        print(fmt(tid, r))
    rep4 = iv.index[0]
else:
    print("  No International, picking other:")
    for tid, r in ov.head(5).iterrows():
        print(fmt(tid, r))
    rep4 = ov.index[0]
print(f"  >>> SELECTED for 42235 replacement: {rep4}")
# FINAL SUMMARY
print(NL + SEP)
print("REPLACEMENT SUMMARY")
print(SEP)
reps = {
    rep1: "HIGH ENERGY replacement for 32437 (Electronic, bad quality) - non-Electronic",
    rep2: "DANCEABLE replacement for 32330 (Pop, bad quality) - non-Hip-Hop",
    rep3: "VOCAL/RAP replacement for 11782 (Hip-Hop, was spoken word) - actual song",
    rep4: "VOCAL replacement for 42235 (International, was interview) - actual song",
}
for tid, reason in reps.items():
    r = df.loc[tid]
    fp = str(tid).zfill(6)
    fp = fp[:3] + "/" + fp + ".mp3"
    print(f"{NL}  Track {tid:>6d}  |  {fp}")
    print(f"    Genre: {r.genre}")
    print(f"    Energy: {r.energy:.3f}  |  Acousticness: {r.acousticness:.3f}  |  Danceability: {r.danceability:.3f}")
    print(f"    Instrumentalness: {r.instrumentalness:.3f}  |  Speechiness: {r.speechiness:.3f}  |  Valence: {r.valence:.3f}  |  Tempo: {r.tempo:.1f}")
    print(f"    WHY: {reason}")

all_12 = KEEP_IDS + [rep1, rep2, rep3, rep4]
print(NL + SEP)
print("FULL LIST OF ALL 12 TRACKS (8 kept + 4 new)")
print(SEP)
fps = []
for tid in sorted(all_12):
    r = df.loc[tid]
    fp = str(tid).zfill(6)
    fp = fp[:3] + "/" + fp + ".mp3"
    fps.append(fp)
    st = "NEW" if tid in reps else "KEEP"
    print(f"  [{st:4s}]  {tid:>6d}  {fp:>14s}  genre={r.genre:<15s}  "
          f"energy={r.energy:.3f}  dance={r.danceability:.3f}  "
          f"speech={r.speechiness:.3f}  instr={r.instrumentalness:.3f}  "
          f"acoustic={r.acousticness:.3f}")

print(NL + SEP)
print("TAR COMMAND (run from fma_small directory):")
print(SEP)
print(NL + "tar -cf listening_test_tracks.tar " + " ".join(fps))

print(NL + SEP)
print("GENRE DIVERSITY CHECK")
print(SEP)
gc = {}
for tid in all_12:
    g = df.loc[tid, "genre"]
    gc[g] = gc.get(g, 0) + 1
for g, c in sorted(gc.items(), key=lambda x: -x[1]):
    print(f"  {g:<15s}: {c}")
print(f"  Total genres: {len(gc)}")