import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

DATA = Path(os.environ.get("HUQIN_DATA", "data"))
AUDIO_ROOT = DATA / "audios" / "SinglePT"
KP_ROOT = DATA / "keypoints"
MEL_ROOT = DATA / "melcache"
SR = 24000
AUDIO_LEN = 3 * SR
KP_LEN = 90
CLASSES = ["DianG", "DuanG", "DunG", "JiG", "PaoG", "Pizz", "Port", "TiaoG", "Tremolo", "Trill", "Vibrato"]


def build_manifest(out_path: Path, test_player=None):
    kp_views = {}
    for view in ["left", "right"]:
        root = next(KP_ROOT.glob(f"*{view}*"))
        kp_views[view] = {
            (p.parts[-3], p.parts[-2], p.stem): p for p in root.glob("Erhu-*/**/*.npz")
        }
    items = []
    for w in sorted(AUDIO_ROOT.glob("Erhu-*/**/*.wav")):
        player, pt, name = w.parts[-3], w.parts[-2], w.stem
        key = (player, pt, name)
        if pt not in CLASSES:
            continue
        if key in kp_views["left"] and key in kp_views["right"]:
            items.append({
                "wav": str(w.relative_to(DATA)),
                "left": str(kp_views["left"][key].relative_to(DATA)),
                "right": str(kp_views["right"][key].relative_to(DATA)),
                "label": pt,
                "player": player,
            })
    rng = random.Random(42)
    splits = {"train": [], "val": [], "test": []}
    if test_player:
        splits["test"] = [it for it in items if it["player"] == test_player]
        rest = [it for it in items if it["player"] != test_player]
        by_class = {}
        for it in rest:
            by_class.setdefault(it["label"], []).append(it)
        for cls, lst in sorted(by_class.items()):
            rng.shuffle(lst)
            n_val = max(1, int(len(lst) * 0.1))
            splits["val"] += lst[:n_val]
            splits["train"] += lst[n_val:]
    else:
        by_class = {}
        for it in items:
            by_class.setdefault(it["label"], []).append(it)
        for cls, lst in sorted(by_class.items()):
            rng.shuffle(lst)
            n = len(lst)
            n_test, n_val = max(1, int(n * 0.1)), max(1, int(n * 0.1))
            splits["test"] += lst[:n_test]
            splits["val"] += lst[n_test:n_test + n_val]
            splits["train"] += lst[n_test + n_val:]
    for lst in splits.values():
        rng.shuffle(lst)
    out_path.write_text(json.dumps(splits))
    print({k: len(v) for k, v in splits.items()},
          "class dist:", Counter(i["label"] for i in splits["train"]))
    return splits


def load_audio(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return fit_audio(wav)


def fit_audio(wav):
    if len(wav) >= AUDIO_LEN:
        s = (len(wav) - AUDIO_LEN) // 2
        wav = wav[s:s + AUDIO_LEN]
    else:
        pad = AUDIO_LEN - len(wav)
        wav = torch.nn.functional.pad(wav, (pad // 2, pad - pad // 2))
    return wav


def load_kp(path, norm=False):
    return prep_kp(np.load(path)["lm"], norm)


def prep_kp(raw, norm=False):
    if norm:
        wrist = raw[:, :, 0:1, :]
        rel = raw - wrist
        scale = np.nanmean(np.linalg.norm(rel, axis=-1), axis=-1, keepdims=True)[..., None]
        rel = rel / np.maximum(scale, 1e-4)
        wrist_vel = np.diff(wrist, axis=0, prepend=wrist[:1])
        lm = np.concatenate(
            [rel.reshape(len(raw), -1), wrist_vel.reshape(len(raw), -1)], axis=1)
    else:
        lm = raw.reshape(len(raw), -1)
    lm = np.nan_to_num(lm, nan=0.0)
    if len(lm) >= KP_LEN:
        s = (len(lm) - KP_LEN) // 2
        lm = lm[s:s + KP_LEN]
    else:
        lm = np.pad(lm, ((0, KP_LEN - len(lm)), (0, 0)))
    vel = np.diff(lm, axis=0, prepend=lm[:1])
    return np.concatenate([lm, vel], axis=1).astype(np.float32)


class ClipDataset(Dataset):
    def __init__(self, items, mode, kp_norm=False):
        self.items = items
        self.mode = mode
        self.kp_norm = kp_norm
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=1024, hop_length=240, n_mels=128)
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        y = CLASSES.index(it["label"])
        out = {"y": y}
        if self.mode in ("audio", "fusion"):
            cache = MEL_ROOT / (DATA / it["wav"]).relative_to(AUDIO_ROOT).with_suffix(".npy")
            if cache.exists():
                out["mel"] = torch.from_numpy(np.load(cache).astype(np.float32)).unsqueeze(0)
            else:
                out["mel"] = self.db(self.mel(load_audio(str(DATA / it["wav"])))).unsqueeze(0)
        if self.mode in ("kp", "fusion"):
            out["kp"] = torch.from_numpy(
                np.concatenate([load_kp(DATA / it["left"], self.kp_norm),
                                load_kp(DATA / it["right"], self.kp_norm)], axis=1))
        return out


class AudioCRNN(nn.Module):
    def __init__(self, emb=256):
        super().__init__()
        chs = [1, 32, 64, 128, 128]
        blocks = []
        for a, b in zip(chs, chs[1:]):
            blocks += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b),
                       nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]
        self.conv = nn.Sequential(*blocks)
        self.gru = nn.GRU(128 * 8, emb // 2, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, mel):
        h = self.conv(mel)
        h = h.permute(0, 3, 1, 2).flatten(2)
        h, _ = self.gru(h)
        return h.mean(1)


class KPNet(nn.Module):
    def __init__(self, in_dim=504, emb=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(256, emb // 2, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, kp):
        h, _ = self.gru(self.proj(kp))
        return h.mean(1)


class Model(nn.Module):
    def __init__(self, mode, n_cls=len(CLASSES), kp_norm=False):
        super().__init__()
        self.mode = mode
        dim = 0
        if mode in ("audio", "fusion"):
            self.audio = AudioCRNN()
            dim += 256
        if mode in ("kp", "fusion"):
            self.kp = KPNet(in_dim=528 if kp_norm else 504)
            dim += 256
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_cls))

    def forward(self, batch):
        embs = []
        if self.mode in ("audio", "fusion"):
            embs.append(self.audio(batch["mel"]))
        if self.mode in ("kp", "fusion"):
            embs.append(self.kp(batch["kp"]))
        return self.head(torch.cat(embs, dim=1))


def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            ys += batch["y"].cpu().tolist()
            ps += logits.argmax(1).cpu().tolist()
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["audio", "kp", "fusion"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-player", default=None)
    ap.add_argument("--kp-norm", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tag = f"_p{args.test_player}" if args.test_player else f"_s{args.seed}"
    if args.kp_norm:
        tag += "_norm"
    out = args.out or DATA / "runs" / f"{args.model}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    split_name = f"splits_{args.test_player}.json" if args.test_player else "splits.json"
    split_file = DATA / "splits" / split_name
    split_file.parent.mkdir(parents=True, exist_ok=True)
    splits = (json.loads(split_file.read_text()) if split_file.exists()
              else build_manifest(split_file, args.test_player))
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders = {
        k: DataLoader(ClipDataset(v, args.model, args.kp_norm), batch_size=args.batch,
                      shuffle=(k == "train"), num_workers=8, pin_memory=True,
                      collate_fn=None)
        for k, v in splits.items()
    }
    counts = Counter(i["label"] for i in splits["train"])
    w = torch.tensor([len(splits["train"]) / (len(CLASSES) * max(counts[c], 1)) for c in CLASSES],
                     dtype=torch.float32, device=device)
    model = Model(args.model, kp_norm=args.kp_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, best_epoch = -1, -1
    for epoch in range(args.epochs):
        model.train()
        tot = 0
        for batch in loaders["train"]:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = crit(model(batch), batch["y"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        ys, ps = run_eval(model, loaders["val"], device)
        f1 = f1_score(ys, ps, average="macro")
        print(f"epoch {epoch} loss {tot / len(loaders['train']):.4f} val_macroF1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), out / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print("early stop", flush=True)
            break

    model.load_state_dict(torch.load(out / "best.pt"))
    ys, ps = run_eval(model, loaders["test"], device)
    report = classification_report(ys, ps, labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(ys, ps, labels=list(range(len(CLASSES))))
    (out / "report.json").write_text(json.dumps({
        "model": args.model,
        "seed": args.seed,
        "test_player": args.test_player,
        "kp_norm": args.kp_norm,
        "test_macro_f1": f1_score(ys, ps, average="macro"),
        "test_micro_f1": f1_score(ys, ps, average="micro"),
        "best_val_macro_f1": best_f1,
        "per_class": report,
    }, indent=2))
    np.savetxt(out / "confusion.csv", cm, fmt="%d", delimiter=",",
               header=",".join(CLASSES))
    print("TEST macro-F1:", f1_score(ys, ps, average="macro"),
          "micro-F1:", f1_score(ys, ps, average="micro"), flush=True)


if __name__ == "__main__":
    main()
