import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import accuracy_score, f1_score

from lbd_train import CLASSES, DATA, SR, Model, fit_audio, prep_kp

EXCERPT_ROOT = DATA / "audios" / "Excerpts"
KP_EXC_ROOT = DATA / "keypoints_excerpts"
LABEL_MAP = {
    "UPort": "Port", "DPort": "Port", "Port": "Port", "DaYin": "Trill", "Trill": "Trill",
    "Vibrato": "Vibrato", "DunG": "DunG", "Pizz": "Pizz", "Tremolo": "Tremolo",
    "PaoG": "PaoG", "TiaoG": "TiaoG", "DianG": "DianG",
}


def find_kp(view, player, piece):
    root = next(KP_EXC_ROOT.glob(f"*{view}*"))
    return next((root / player).rglob(f"{piece}.npz"))


def load_models(runs, model, norm, seeds, device):
    out = []
    for s in seeds:
        m = Model(model, kp_norm=norm).to(device).eval()
        m.load_state_dict(torch.load(runs / f"{model}_s{s}{'_norm' if norm else ''}" / "best.pt", map_location=device))
        out.append(m)
    return out


def notes(pt_csv):
    with pt_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = LABEL_MAP.get(row.get("PT1-2", "").strip())
            if label:
                yield float(row["onset"]), float(row["duration"]), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--min-len", type=float, default=0.6)
    ap.add_argument("--out", type=Path, default=DATA / "demo" / "excerpt_predictions.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_models = load_models(args.runs, "audio", False, args.seeds, device)
    fusion_models = load_models(args.runs, "fusion", True, args.seeds, device)
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=240, n_mels=128)
    db = torchaudio.transforms.AmplitudeToDB()

    records = []
    for pt_csv in sorted(EXCERPT_ROOT.glob("Erhu-*/*/*-PT.csv")):
        player, piece = pt_csv.parents[1].name, pt_csv.parent.name
        wav, sr = torchaudio.load(str(pt_csv.with_name(pt_csv.name.removesuffix("-PT.csv") + ".wav")))
        wav = torchaudio.functional.resample(wav.mean(0), sr, SR)
        kp = {v: np.load(find_kp(v, player, piece)) for v in ["left", "right"]}
        fps = float(kp["left"]["fps"])
        for onset, dur, label in notes(pt_csv):
            span = max(dur, args.min_len)
            a = max(onset + dur / 2 - span / 2, 0.0)
            seg = fit_audio(wav[int(a * SR):int((a + span) * SR)])
            batch = {"mel": db(mel(seg)).unsqueeze(0).unsqueeze(0).to(device)}
            f0, f1 = int(round(a * fps)), max(int(round((a + span) * fps)), int(round(a * fps)) + 1)
            batch["kp"] = torch.from_numpy(np.concatenate(
                [prep_kp(kp[v]["lm"][f0:f1], True) for v in ["left", "right"]], axis=1)).unsqueeze(0).to(device)
            with torch.no_grad():
                pa = torch.stack([m(batch).softmax(1) for m in audio_models]).mean(0)[0].cpu().numpy()
                pf = torch.stack([m(batch).softmax(1) for m in fusion_models]).mean(0)[0].cpu().numpy()
            records.append({
                "player": player, "piece": piece, "onset": onset, "duration": dur, "label": label,
                "audio": CLASSES[int(pa.argmax())], "fusion": CLASSES[int(pf.argmax())],
                "audio_conf": float(pa.max()), "fusion_conf": float(pf.max()),
            })
        print(player, piece, sum(r["piece"] == piece and r["player"] == player for r in records), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    y = [r["label"] for r in records]
    labels = sorted(set(y))
    metrics = {k: {"accuracy": accuracy_score(y, [r[k] for r in records]),
                   "macro_f1": f1_score(y, [r[k] for r in records], labels=labels, average="macro", zero_division=0)}
               for k in ["audio", "fusion"]}
    args.out.write_text(json.dumps({"metrics": metrics, "n_notes": len(records), "notes": records}, indent=1))
    print(json.dumps(metrics, indent=2), "notes:", len(records))


if __name__ == "__main__":
    main()
