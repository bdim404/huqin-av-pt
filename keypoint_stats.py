import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from lbd_train import DATA


def summarise(files, by_class):
    rows = [r for f in files for r in csv.DictReader(open(f))]
    frames = np.array([int(r["frames"]) for r in rows])
    rate = np.array([float(r["detect_rate"]) for r in rows])
    low = [r["video"].split("/")[-2] for r in rows if float(r["detect_rate"]) < 0.5]
    return {
        "videos": len(rows),
        "frames": int(frames.sum()),
        "frame_level_detection_rate": round(float((frames * rate).sum() / frames.sum()), 4),
        "median_clip_detection_rate": float(np.median(rate)),
        "videos_below_50pct": int((rate < 0.5).sum()),
        "below_50pct_by_class": dict(collections.Counter(low)) if by_class else None,
        "definition": "a frame counts as detected if MediaPipe finds at least one hand in that camera view",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/keypoint_stats.json"))
    args = ap.parse_args()
    out = {
        "singlept": summarise([DATA / f"kp_manifest_{v}.csv" for v in ["left", "right"]], True),
        "excerpts": summarise([DATA / f"kp_exc_manifest_{v}.csv" for v in ["left", "right"]], False),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
