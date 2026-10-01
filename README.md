# Hands Tell the Technique

Code, data splits, results and model weights for *Hands Tell the Technique: Audio-Visual Playing Technique Classification on CCOM-HuQin* (Jimeng Tian, ISMIR 2026 Late-Breaking/Demo).

A log-mel CRNN is fused with a BiGRU over MediaPipe hand landmarks from the left- and right-hand close-up cameras of [CCOM-HuQin](https://doi.org/10.5281/zenodo.6957454). On 11-class erhu playing technique classification, fusion with wrist-centred, scale-normalised hand geometry reaches 98.4 ± 0.1 macro-F1, compared with 96.2 ± 0.8 for audio only (3 seeds).

| Model | Random split (3 seeds) | Player held out (3 folds, seed 42) |
|---|---|---|
| Audio CRNN | 96.2 ± 0.8 | 55.9 ± 4.0 |
| Keypoints, raw | 76.7 ± 1.2 | 19.1 ± 6.7 |
| Keypoints, normalised | 93.9 ± 0.2 | 43.2 ± 2.6 |
| Fusion, raw | 96.8 ± 0.5 | 52.9 ± 4.5 |
| Fusion, normalised | 98.4 ± 0.1 | 51.8 ± 6.8 |

## Contents

- `lbd_train.py` trains and evaluates one model (`--model audio|kp|fusion`, `--kp-norm`, `--seed`, `--test-player`).
- `precompute_mels.py`, `extract_hand_keypoints.py` compute log-mel and MediaPipe hand-landmark features.
- `demo_excerpts.py` classifies the PT-annotated notes of the 21 erhu excerpts with 3-seed ensembles; `render_clips.py` and `render_demo.py` render the demo video; `make_figures.py` draws the poster figures from `results/`.
- `pipeline.sh` runs everything above end to end.
- `build_erhu_manifest.py` writes dataset statistics (clip and note counts per performer and technique).
- `splits/` holds the exact train/val/test splits (random split and the three leave-one-player-out folds), with paths relative to the data root.
- `results/runs/` holds the test report and confusion matrix of all 30 runs; `results/demo/pred_min0.6.json` the note-level excerpt predictions; `results/keypoint_stats.json` the hand-detection statistics.
- Model weights (`best.pt` of all 30 runs) are attached to the [v1.0 release](https://github.com/bdim404/huqin-av-pt/releases/tag/v1.0); unpack them into `$HUQIN_DATA/runs/`.

## Setup

Python 3.11 with the packages in `requirements.txt` (PyTorch 2.6 / CUDA 12.4 was used; any recent CUDA build works). `render_*.py` also need `ffmpeg` on `PATH` and the Fira Sans OTF fonts.

Download CCOM-HuQin v2.0.1 from Zenodo: the audio archive, the `left` and `right` SinglePT and Excerpts video archives, and `front`-Excerpts for the demo. Extract the audio archive to `$HUQIN_DATA/audios` and pass the directory holding the video zips as `ZIPS` (only the erhu folders are extracted):

```
data/
  audios/            SinglePT/, Excerpts/ from CCOM-HuQin-v2.0.1-audios
  videos/            CCOM-HuQin-v2.0-videos-{left,right}-SinglePT/Erhu-*
  videos_excerpts/   video-{left,right}/Erhu-*, CCOM-HuQin-v2.0-videos-front-Excerpts/Erhu-*
```

`hand_landmarker.task` is the MediaPipe Hand Landmarker model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Running

```
export HUQIN_DATA=/path/to/data HAND_LANDMARKER=/path/to/hand_landmarker.task
ZIPS=/path/to/zips bash pipeline.sh
```

or step by step:

```
python precompute_mels.py
for v in left right; do
  python extract_hand_keypoints.py $HUQIN_DATA/videos/CCOM-HuQin-v2.0-videos-$v-SinglePT $HUQIN_DATA/keypoints/CCOM-HuQin-v2.0-videos-$v-SinglePT
  python extract_hand_keypoints.py $HUQIN_DATA/videos_excerpts/video-$v $HUQIN_DATA/keypoints_excerpts/video-$v
done
python lbd_train.py --model fusion --kp-norm --seed 42
python lbd_train.py --model audio --test-player Erhu-1
python demo_excerpts.py --out $HUQIN_DATA/demo/pred_min0.6.json
python render_clips.py --font-dir /path/to/fira-sans
python render_demo.py --player Erhu-1 --piece Henan_Xiao_Qu-part1 --start 8 --end 40 --pred $HUQIN_DATA/demo/pred_min0.6.json --font-dir /path/to/fira-sans --out excerpt_demo.mp4
python make_figures.py results/runs --font-dir /path/to/fira-sans
```

`lbd_train.py` reads splits from `splits/` (override with `HUQIN_SPLITS`) and writes runs to `$HUQIN_DATA/runs/`.

## License

Code: MIT (see `LICENSE`). The splits, results and model weights are derived from CCOM-HuQin and are released under its CC BY-NC-SA 4.0 license.

## Citation

```
@inproceedings{tian2026hands,
  author    = {Jimeng Tian},
  title     = {Hands Tell the Technique: Audio-Visual Playing Technique Classification on {CCOM-HuQin}},
  booktitle = {Extended Abstracts for the Late-Breaking Demo Session of the 27th Int. Society for Music Information Retrieval Conf. (ISMIR)},
  year      = {2026}
}
```
