# Hands Tell the Technique

Code for *Hands Tell the Technique: Audio-Visual Playing Technique Classification on CCOM-HuQin* (Jimeng Tian, ISMIR 2026 Late-Breaking/Demo).

An audio-only log-mel CRNN is fused with a BiGRU over MediaPipe hand landmarks from the left- and right-hand close-up cameras of [CCOM-HuQin](https://doi.org/10.5281/zenodo.6957454). On 11-class erhu playing technique classification, fusion with wrist-centred, scale-normalised hand geometry reaches 98.4 ± 0.1 macro-F1, compared with 96.2 ± 0.8 for audio only (3 seeds). Per-run reports and confusion matrices are in `results/runs/`; note-level predictions on the 21 real erhu excerpts are in `results/demo/`.

## Data

Download CCOM-HuQin v2.0.1 (CC BY-NC-SA 4.0) from Zenodo: the audio archive and the `left`/`right` SinglePT and Excerpts video archives (`front`-Excerpts too for the demo). Lay them out under one data root:

```
data/
  audios/                      CCOM-HuQin-v2.0.1-audios (SinglePT/, Excerpts/)
  videos/                      CCOM-HuQin-v2.0-videos-{left,right}-SinglePT/Erhu-*
  videos_excerpts/             video-{left,right}/Erhu-*, CCOM-HuQin-v2.0-videos-front-Excerpts/Erhu-*
```

## Running

```
export HUQIN_DATA=/path/to/data HAND_LANDMARKER=/path/to/hand_landmarker.task
python precompute_mels.py
python extract_hand_keypoints.py $HUQIN_DATA/videos/CCOM-HuQin-v2.0-videos-left-SinglePT $HUQIN_DATA/keypoints/CCOM-HuQin-v2.0-videos-left-SinglePT
python lbd_train.py --model fusion --kp-norm --seed 42
python lbd_train.py --model audio --test-player Erhu-1
python demo_excerpts.py --out $HUQIN_DATA/demo/pred_min0.6.json
python render_clips.py --font-dir /path/to/fira-sans
python render_demo.py --player Erhu-1 --piece Henan_Xiao_Qu-part1 --start 8 --end 40 --pred $HUQIN_DATA/demo/pred_min0.6.json --font-dir /path/to/fira-sans --out excerpt_demo.mp4
python make_figures.py $HUQIN_DATA/runs --font-dir /path/to/fira-sans
```

`pipeline.sh` runs the full set of experiments in the paper (3 seeds × 5 models on the random split, 3 folds × 5 models leave-one-player-out). The exact train/val/test splits are in `splits/`.

`hand_landmarker.task` is the MediaPipe Hand Landmarker model: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Citation

```
@inproceedings{tian2026hands,
  author    = {Jimeng Tian},
  title     = {Hands Tell the Technique: Audio-Visual Playing Technique Classification on {CCOM-HuQin}},
  booktitle = {Extended Abstracts for the Late-Breaking Demo Session of the 27th Int. Society for Music Information Retrieval Conf. (ISMIR)},
  year      = {2026}
}
```
