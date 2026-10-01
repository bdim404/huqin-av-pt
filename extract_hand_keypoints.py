import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = os.environ.get("HAND_LANDMARKER", "hand_landmarker.task")


def make_landmarker():
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def process_video(args):
    video_path, out_path = args
    import mediapipe as mpi

    try:
        landmarker = make_landmarker()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return (str(video_path), -1, 0, "cannot_open")
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        lm_frames = []
        handed_frames = []
        conf_frames = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            image = mpi.Image(
                image_format=mpi.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(image, int(idx / fps * 1000))
            lm = np.full((2, 21, 3), np.nan, dtype=np.float32)
            handed = np.zeros(2, dtype=np.int8)
            conf = np.zeros(2, dtype=np.float32)
            for h, (hand_lms, handedness) in enumerate(
                zip(result.hand_landmarks[:2], result.handedness[:2])
            ):
                lm[h] = [[p.x, p.y, p.z] for p in hand_lms]
                cat = handedness[0]
                handed[h] = 1 if cat.category_name == "Left" else 2
                conf[h] = cat.score
            lm_frames.append(lm)
            handed_frames.append(handed)
            conf_frames.append(conf)
            idx += 1
        cap.release()
        landmarker.close()
        if not lm_frames:
            return (str(video_path), -1, 0, "no_frames")
        lm_arr = np.stack(lm_frames)
        detect_rate = float(np.mean(~np.isnan(lm_arr[:, 0, 0, 0]) | ~np.isnan(lm_arr[:, 1, 0, 0])))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            lm=lm_arr,
            handed=np.stack(handed_frames),
            conf=np.stack(conf_frames),
            fps=np.float32(fps),
        )
        return (str(video_path), idx, detect_rate, "")
    except Exception as e:
        return (str(video_path), -1, 0, f"{type(e).__name__}:{e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--manifest", type=Path, default=Path("keypoints_manifest.csv"))
    args = parser.parse_args()

    videos = sorted(args.video_dir.rglob("*.mp4"))
    jobs = []
    for v in videos:
        rel = v.relative_to(args.video_dir)
        out = args.out_dir / rel.with_suffix(".npz")
        if not out.exists():
            jobs.append((v, out))
    print(f"total {len(videos)} videos, {len(jobs)} to process", flush=True)

    write_header = not args.manifest.exists()
    done = 0
    with open(args.manifest, "a") as mf:
        if write_header:
            mf.write("video,frames,detect_rate,error\n")
        with mp.Pool(args.workers) as pool:
            for path, frames, rate, err in pool.imap_unordered(process_video, jobs, chunksize=4):
                done += 1
                mf.write(f"{path},{frames},{rate:.4f},{err}\n")
                mf.flush()
                if done % 50 == 0 or err:
                    print(f"[{done}/{len(jobs)}] rate={rate:.2%} {'ERR:' + err if err else ''} {Path(path).name}", flush=True)
    print(f"finished {done} videos", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    sys.exit(main())
