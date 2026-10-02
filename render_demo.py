import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lbd_train import DATA

VIDEO_EXC_ROOT = DATA / "videos_excerpts"
KP_EXC_ROOT = DATA / "keypoints_excerpts"
EXCERPT_ROOT = DATA / "audios" / "Excerpts"
W, H = 1920, 1080
INK, INK2, RULE, PANEL, SURFACE = "#0b0b0b", "#52514e", "#d9d8d3", "#f4f3ef", "#ffffff"
GOOD, BAD, FUSION, AUDIO = "#0ca30c", "#d03b3b", "#2a78d6", "#eb6834"
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
WINDOW = 3.0
ABBR = {"DianG": "Dia", "DuanG": "Dua", "DunG": "Dun", "JiG": "Ji", "PaoG": "Pao", "Pizz": "Piz", "Port": "Por",
        "TiaoG": "Tia", "Tremolo": "Tre", "Trill": "Tri", "Vibrato": "Vib"}


def find(root, view, player, piece, ext):
    return next((next(root.glob(f"*{view}*")) / player).rglob(f"{piece}{ext}"))


def tint(hex_color, a):
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)])
    return tuple(int(v) for v in (c * a + 255 * (1 - a)).round())


def draw_icon(d, x, y, ok, s=20):
    if ok:
        d.line([(x, y + s * 0.55), (x + s * 0.38, y + s), (x + s, y)], fill=GOOD, width=5, joint="curve")
    else:
        d.line([(x, y), (x + s, y + s)], fill=BAD, width=5)
        d.line([(x + s, y), (x, y + s)], fill=BAD, width=5)


def fit(frame, w, h):
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    img = cv2.resize(frame, (int(fw * s), int(fh * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 11, dtype=np.uint8)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    return canvas, (x0, y0, s * fw, s * fh)


def overlay_hands(d, lm, box, ox, oy):
    x0, y0, w, h = box
    for hand in lm:
        if np.isnan(hand[0, 0]):
            continue
        pts = [(ox + x0 + p[0] * w, oy + y0 + p[1] * h) for p in hand]
        for a, b in HAND_EDGES:
            d.line([pts[a], pts[b]], fill="#ffffff", width=3)
        for p in pts:
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill="#1baf7a", outline="#ffffff", width=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True)
    ap.add_argument("--piece", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--pred", type=Path, default=DATA / "demo" / "excerpt_predictions.json")
    ap.add_argument("--font-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--banner", default="Hands Tell the Technique  ·  ISMIR 2026 LBD  ·  github.com/bdim404/huqin-av-pt")
    args = ap.parse_args()

    f = {k: ImageFont.truetype(str(args.font_dir / f"FiraSans-{w}.otf"), s) for k, (w, s) in {
        "lane": ("Regular", 26), "seg": ("Medium", 21), "now": ("Regular", 30), "nowb": ("Bold", 30),
        "cap": ("Regular", 22), "tag": ("Medium", 24)}.items()}
    notes = [n for n in json.loads(args.pred.read_text())["notes"]
             if n["player"] == args.player and n["piece"] == args.piece]
    caps = {v: cv2.VideoCapture(str(find(VIDEO_EXC_ROOT, v, args.player, args.piece, ".mp4")))
            for v in ["front", "left", "right"]}
    kps = {v: np.load(find(KP_EXC_ROOT, v, args.player, args.piece, ".npz"))["lm"] for v in ["left", "right"]}
    fps = caps["front"].get(cv2.CAP_PROP_FPS) or 29.97
    f0, f1 = int(args.start * fps), int(args.end * fps)
    vfps = {v: c.get(cv2.CAP_PROP_FPS) or fps for v, c in caps.items()}
    kfps = {v: float(np.load(find(KP_EXC_ROOT, v, args.player, args.piece, ".npz"))["fps"]) for v in kps}
    pos = {v: 0 for v in caps}
    wav = next((EXCERPT_ROOT / args.player).rglob(f"{args.piece}.wav"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", f"{fps}", "-i", "-", "-ss", f"{args.start}", "-t", f"{args.end - args.start}", "-i", str(wav),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", str(args.out)], stdin=subprocess.PIPE)

    top_h, front_w = 700, 1244
    hand_w, hand_h = W - front_w - 12, (top_h - 12) // 2
    lanes = [("Ground truth", "label", INK2), ("Audio only", "audio", AUDIO), ("Audio + hands", "fusion", FUSION)]
    tl_x0, tl_x1, lane_y0, lane_h = 300, W - 40, 818, 62
    for fi in range(f0, f1):
        t = fi / fps
        frames = {}
        for v, c in caps.items():
            tgt = int(round(t * vfps[v]))
            while pos[v] < tgt:
                c.grab()
                pos[v] += 1
            ok, fr = c.read()
            pos[v] += 1
            frames[v] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else np.zeros((720, 1280, 3), np.uint8)
        img = Image.new("RGB", (W, H), SURFACE)
        fr, _ = fit(frames["front"], front_w, top_h)
        img.paste(Image.fromarray(fr), (0, 0))
        d = ImageDraw.Draw(img)
        for k, v in enumerate(["left", "right"]):
            fr, box = fit(frames[v], hand_w, hand_h)
            ox, oy = front_w + 12, k * (hand_h + 12)
            img.paste(Image.fromarray(fr), (ox, oy))
            lm = kps[v][min(int(round(t * kfps[v])), len(kps[v]) - 1)]
            overlay_hands(d, lm, box, ox, oy)
            d.rounded_rectangle([ox + 14, oy + 14, ox + 210, oy + 54], 8, fill=(11, 11, 11))
            d.text((ox + 26, oy + 20), f"{'Left' if v == 'left' else 'Right'}-hand camera", font=f["tag"], fill="#ffffff")

        cur = next((n for n in notes if n["onset"] <= t < n["onset"] + n["duration"]), None)
        x = 40
        d.text((x, 728), "Now:", font=f["now"], fill=INK2)
        x += 90
        if cur:
            for lab, key, _ in lanes:
                d.text((x, 728), lab, font=f["now"], fill=INK2)
                x += d.textlength(lab, font=f["now"]) + 14
                d.text((x, 728), cur[key], font=f["nowb"], fill=INK)
                x += d.textlength(cur[key], font=f["nowb"]) + 10
                if key != "label":
                    draw_icon(d, x, 736, cur[key] == cur["label"])
                    x += 30
                x += 46
        else:
            d.text((x, 728), "no technique annotated", font=f["now"], fill=INK2)

        def tx(s):
            return tl_x0 + (s - (t - WINDOW)) / (2 * WINDOW) * (tl_x1 - tl_x0)

        for li, (lab, key, color) in enumerate(lanes):
            y = lane_y0 + li * (lane_h + 10)
            d.rectangle([tl_x0, y, tl_x1, y + lane_h], fill=PANEL)
            d.rectangle([40, y + 14, 48, y + lane_h - 14], fill=color)
            d.text((62, y + 15), lab, font=f["lane"], fill=INK)
            for n in notes:
                a, b = tx(n["onset"]), tx(n["onset"] + n["duration"])
                if b < tl_x0 or a > tl_x1:
                    continue
                a, b = max(a, tl_x0), min(b, tl_x1)
                if b - a < 4:
                    continue
                if key == "label":
                    fill, edge = tint(INK2, 0.18), None
                else:
                    ok = n[key] == n["label"]
                    fill, edge = tint(GOOD if ok else BAD, 0.22), GOOD if ok else BAD
                d.rectangle([a + 1, y + 4, b - 1, y + lane_h - 4], fill=fill)
                if edge:
                    d.rectangle([a + 1, y + lane_h - 8, b - 1, y + lane_h - 4], fill=edge)
                lab = ABBR[n[key]]
                if b - a > d.textlength(lab, font=f["seg"]) + 4:
                    d.text(((a + b - d.textlength(lab, font=f["seg"])) / 2, y + 16), lab, font=f["seg"], fill=INK)
        px = tx(t)
        d.line([(px, lane_y0 - 8), (px, lane_y0 + 3 * (lane_h + 10))], fill=INK, width=3)
        d.text((40, H - 40), args.banner, font=f["cap"], fill=INK2)
        d.text((W - 40 - d.textlength(f"{args.piece.replace('_', ' ')}, {args.player}  ·  video: CCOM-HuQin, CC BY-NC-SA 4.0", font=f["cap"]), H - 40),
               f"{args.piece.replace('_', ' ')}, {args.player}  ·  video: CCOM-HuQin, CC BY-NC-SA 4.0", font=f["cap"], fill=INK2)
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
