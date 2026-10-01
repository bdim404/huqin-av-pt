import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from lbd_train import CLASSES, DATA, SPLIT_DIR, ClipDataset, Model
from render_demo import INK, INK2, SURFACE, draw_icon, fit, overlay_hands

W, H = 1920, 1080
SLOW = 3
LOOPS = 2
HOLD = 2.5


def ensemble(items, mode, norm, seeds, runs, device):
    loader = DataLoader(ClipDataset(items, mode, norm), batch_size=64, num_workers=8)
    probs = 0
    for s in seeds:
        m = Model(mode, kp_norm=norm).to(device).eval()
        m.load_state_dict(torch.load(runs / f"{mode}_s{s}{'_norm' if norm else ''}" / "best.pt", map_location=device))
        out = []
        with torch.no_grad():
            for batch in loader:
                out.append(m({k: v.to(device) for k, v in batch.items()}).softmax(1).cpu())
        probs = probs + torch.cat(out)
    return (probs / len(seeds)).numpy()


def video_of(npz):
    p = DATA / npz
    rel = p.relative_to(DATA / "keypoints")
    return DATA / "videos" / rel.with_suffix(".mp4")


def pick(items, pa, pf, n):
    fixed = [i for i, it in enumerate(items)
             if CLASSES[pa[i].argmax()] != it["label"] and CLASSES[pf[i].argmax()] == it["label"]]
    fixed.sort(key=lambda i: (not (items[i]["label"] == "DunG" and CLASSES[pa[i].argmax()] == "TiaoG"), -pf[i].max()))
    chosen, seen = [], set()
    for i in fixed:
        key = (items[i]["label"], CLASSES[pa[i].argmax()])
        if key in seen and len(set(k for k, _ in seen)) < n:
            continue
        seen.add(key)
        chosen.append(i)
        if len(chosen) == n:
            break
    return chosen


def render(item, pa, pf, k, n, fonts, out):
    caps = {v: cv2.VideoCapture(str(video_of(item[v]))) for v in ["left", "right"]}
    kps = {v: np.load(DATA / item[v])["lm"] for v in ["left", "right"]}
    fps = caps["left"].get(cv2.CAP_PROP_FPS) or 29.97
    frames = []
    while True:
        got = {}
        for v, c in caps.items():
            ok, fr = c.read()
            if ok:
                got[v] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if len(got) < 2:
            break
        frames.append(got)
    rows = [("Ground truth", item["label"], None), ("Audio only", CLASSES[pa.argmax()], pa.max()),
            ("Audio + hands", CLASSES[pf.argmax()], pf.max())]
    hold = int(HOLD * fps)
    dur = (len(frames) * SLOW * LOOPS + hold) / fps
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", f"{fps}", "-i", "-", "-i", str(DATA / item["wav"]), "-filter_complex",
        f"[1:a]apad,atrim=0:{len(frames) / fps},asetpts=N/SR/TB,aloop=loop={LOOPS - 1}:size=2e9,"
        f"atempo=0.5,atempo={2 / SLOW},apad,atrim=0:{dur},aformat=sample_rates=48000:channel_layouts=stereo[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{dur}", str(out)], stdin=subprocess.PIPE)
    seq = [(i, False) for _ in range(LOOPS) for i in range(len(frames)) for _ in range(SLOW)] + [(len(frames) - 1, True)] * hold
    for i, reveal in seq:
        img = Image.new("RGB", (W, H), SURFACE)
        d = ImageDraw.Draw(img)
        d.text((40, 30), f"Held-out test clip {k}/{n}", font=fonts["head"], fill=INK)
        d.text((40, 84), f"played twice at 1/{SLOW} speed  ·  {item['player']}", font=fonts["cap"], fill=INK2)
        for j, v in enumerate(["left", "right"]):
            fr, box = fit(frames[i][v], 912, 513)
            ox, oy = 40 + j * 928, 140
            img.paste(Image.fromarray(fr), (ox, oy))
            overlay_hands(d, kps[v][min(i, len(kps[v]) - 1)], box, ox, oy)
            d.rounded_rectangle([ox + 14, oy + 14, ox + 210, oy + 54], 8, fill=(11, 11, 11))
            d.text((ox + 26, oy + 20), f"{'Left' if v == 'left' else 'Right'}-hand camera", font=fonts["tag"], fill="#ffffff")
        y = 690
        for lab, pred, conf in rows:
            d.text((40, y), lab, font=fonts["row"], fill=INK2)
            if conf is None or reveal:
                d.text((380, y), pred, font=fonts["rowb"], fill=INK)
                if conf is not None:
                    x = 380 + d.textlength(pred, font=fonts["rowb"]) + 18
                    draw_icon(d, x, y + 12, pred == item["label"], s=28)
                    d.text((x + 52, y + 4), f"{conf:.0%} confidence", font=fonts["cap"], fill=INK2)
            y += 92
        d.text((40, H - 40), "Hands Tell the Technique  ·  ISMIR 2026 LBD  ·  github.com/bdim404/huqin-av-pt",
               font=fonts["cap"], fill=INK2)
        cred = "video: CCOM-HuQin, CC BY-NC-SA 4.0"
        d.text((W - 40 - d.textlength(cred, font=fonts["cap"]), H - 40), cred, font=fonts["cap"], fill=INK2)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=DATA / "runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--font-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DATA / "demo" / "clips")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = json.loads((SPLIT_DIR / "splits.json").read_text())["test"]
    pa = ensemble(items, "audio", False, args.seeds, args.runs, device)
    pf = ensemble(items, "fusion", True, args.seeds, args.runs, device)
    print("ensemble acc audio", np.mean([CLASSES[p.argmax()] == it["label"] for p, it in zip(pa, items)]),
          "fusion", np.mean([CLASSES[p.argmax()] == it["label"] for p, it in zip(pf, items)]))
    fonts = {k: ImageFont.truetype(str(args.font_dir / f"FiraSans-{w}.otf"), s) for k, (w, s) in {
        "head": ("Bold", 44), "cap": ("Regular", 24), "tag": ("Medium", 24), "row": ("Regular", 44),
        "rowb": ("Bold", 44)}.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = pick(items, pa, pf, args.n)
    for k, i in enumerate(chosen, 1):
        it = items[i]
        print(k, it["label"], "audio:", CLASSES[pa[i].argmax()], "fusion:", CLASSES[pf[i].argmax()], it["wav"])
        render(it, pa[i], pf[i], k, len(chosen), fonts, args.out / f"clip{k}.mp4")


if __name__ == "__main__":
    main()
