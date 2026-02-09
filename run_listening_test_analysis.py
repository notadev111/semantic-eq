"""
Run semantic analysis on all 12 listening test clips.
Produces per-clip figures and a summary comparison table.
"""
import subprocess
import json
import sys
from pathlib import Path

# All 12 clips with their metadata
clips = [
    {"id": "001544", "path": "listening_test_tracks/001/001544.mp3", "genre": "Folk", "category": "Quiet, acoustic"},
    {"id": "006367", "path": "listening_test_tracks/006/006367.mp3", "genre": "International", "category": "Vocal song"},
    {"id": "014542", "path": "listening_test_tracks/014/014542.mp3", "genre": "Hip-Hop", "category": "Rap vocals"},
    {"id": "014570", "path": "listening_test_tracks/014/014570.mp3", "genre": "Instrumental", "category": "Quiet, acoustic, instrumental"},
    {"id": "028802", "path": "listening_test_tracks/028/028802.mp3", "genre": "Pop", "category": "High energy pop"},
    {"id": "030486", "path": "listening_test_tracks/030/030486.mp3", "genre": "Pop", "category": "Danceable pop"},
    {"id": "040242", "path": "listening_test_tracks/040/040242.mp3", "genre": "Folk", "category": "Instrumental"},
    {"id": "048293", "path": "listening_test_tracks/048/048293.mp3", "genre": "Electronic", "category": "Instrumental electronic"},
    {"id": "071508", "path": "listening_test_tracks/071/071508.mp3", "genre": "Electronic", "category": "Bright electronic"},
    {"id": "089846", "path": "listening_test_tracks/089/089846.mp3", "genre": "Hip-Hop", "category": "Danceable"},
    {"id": "115761", "path": "listening_test_tracks/115/115761.mp3", "genre": "Rock", "category": "Max energy rock"},
    {"id": "115765", "path": "listening_test_tracks/115/115765.mp3", "genre": "Rock", "category": "Loud non-acoustic rock"},
]

# Usable semantic terms (excluding re27, test, brighter which are meaningless/redundant)
USABLE_TERMS = ["warm", "bright", "thin", "full", "muddy", "clear", "airy", "deep", "boomy", "tinny"]

def run_analysis():
    results = {}

    for i, clip in enumerate(clips, 1):
        clip_id = clip["id"]
        audio_path = clip["path"]
        output_path = f"results/listening_test/clip_{clip_id}.png"

        print(f"\n{'='*60}")
        print(f"Clip {i:02d}/12: {clip_id} ({clip['genre']} - {clip['category']})")
        print(f"{'='*60}")

        # Run analysis script and capture output
        cmd = [
            sys.executable, "analyze_audio_semantic.py",
            "--audio", audio_path,
            "--output", output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Parse the output to extract scores
        scores = {}
        lines = result.stdout.split('\n')
        in_results = False
        for line in lines:
            if "TOP SEMANTIC MATCHES" in line:
                in_results = True
                continue
            if in_results and "TEMPORAL" in line:
                in_results = False
                break
            if in_results:
                # Parse lines like "   1. warm            0.5831 ± 0.0202  (stable)"
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].rstrip('.').isdigit():
                    term = parts[1]
                    try:
                        score = float(parts[2])
                        scores[term] = score
                    except ValueError:
                        pass

        results[clip_id] = {
            "genre": clip["genre"],
            "category": clip["category"],
            "scores": scores
        }

        # Print top usable terms for this clip
        usable = {k: v for k, v in scores.items() if k in USABLE_TERMS}
        sorted_terms = sorted(usable.items(), key=lambda x: -x[1])
        top3 = sorted_terms[:3]
        print(f"  Model top 3: {', '.join(f'{t}({s:.2f})' for t, s in top3)}")

        if result.returncode != 0:
            print(f"  WARNING: Script exited with code {result.returncode}")
            if result.stderr:
                # Only print last few lines of error
                err_lines = result.stderr.strip().split('\n')[-3:]
                for l in err_lines:
                    print(f"  ERR: {l}")

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: MODEL PREDICTIONS FOR LISTENING TEST")
    print(f"{'='*80}")
    print(f"\n{'Clip':<8} {'Genre':<14} {'Category':<25} {'Top 3 Descriptors'}")
    print("-" * 80)

    for i, clip in enumerate(clips, 1):
        cid = clip["id"]
        r = results.get(cid, {})
        scores = r.get("scores", {})
        usable = {k: v for k, v in scores.items() if k in USABLE_TERMS}
        sorted_terms = sorted(usable.items(), key=lambda x: -x[1])
        top3_str = ", ".join(f"{t}({s:.2f})" for t, s in sorted_terms[:3])
        print(f"Clip {i:02d}  {clip['genre']:<14} {clip['category']:<25} {top3_str}")

    # Save results to JSON
    output_json = Path("results/listening_test/model_predictions.json")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved predictions to: {output_json}")

    # Print the form-ready comparison table
    print(f"\n\n{'='*80}")
    print("FORM ANSWER KEY (what model predicts vs what humans will pick)")
    print(f"{'='*80}\n")

    for i, clip in enumerate(clips, 1):
        cid = clip["id"]
        r = results.get(cid, {})
        scores = r.get("scores", {})
        usable = {k: v for k, v in scores.items() if k in USABLE_TERMS}
        sorted_terms = sorted(usable.items(), key=lambda x: -x[1])

        print(f"Clip {i:02d} ({clip['genre']} - {clip['category']}):")
        for term, score in sorted_terms:
            bar = '#' * int(score * 30)
            print(f"  {term:<10} {score:.3f} {bar}")
        print()


if __name__ == "__main__":
    run_analysis()
