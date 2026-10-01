import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

matplotlib.use("Agg")

CLASSES = ["DianG", "DuanG", "DunG", "JiG", "PaoG", "Pizz", "Port", "TiaoG", "Tremolo", "Trill", "Vibrato"]
INK, INK2, RULE, SURFACE = "#0b0b0b", "#52514e", "#d9d8d3", "#ffffff"
AUDIO, VISUAL, FUSION = "#eb6834", "#1baf7a", "#2a78d6"
PLAYERS = ["#4a3aa7", "#e87ba4", "#008300"]
MODELS = [
    ("audio", "", "Audio CRNN", AUDIO),
    ("kp", "", "Keypoints, raw", VISUAL),
    ("kp", "_norm", "Keypoints, normalised", VISUAL),
    ("fusion", "", "Fusion, raw", FUSION),
    ("fusion", "_norm", "Fusion, normalised", FUSION),
]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


def setup_style(font_dir):
    for f in Path(font_dir).glob("FiraSans-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Fira Sans",
        "font.size": 26,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK2,
        "axes.linewidth": 1.2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 3,
    })


def load_runs(runs):
    out = {}
    for r in runs.glob("*/report.json"):
        rep = json.loads(r.read_text())
        key = (rep["model"], "_norm" if rep["kp_norm"] else "",
               "player" if rep["test_player"] else "random")
        cm = np.loadtxt(r.parent / "confusion.csv", delimiter=",", skiprows=1)
        out.setdefault(key, []).append({"rep": rep, "cm": cm})
    return out


def summary(runs):
    rows = []
    for model, norm, label, _ in MODELS:
        row = {"model": label}
        for proto in ["random", "player"]:
            f1 = [x["rep"]["test_macro_f1"] * 100 for x in runs.get((model, norm, proto), [])]
            row[proto] = (float(np.mean(f1)), float(np.std(f1)), len(f1)) if f1 else None
        rows.append(row)
    return rows


def per_class_f1(runs, model, norm):
    vals = np.array([[x["rep"]["per_class"][c]["f1-score"] * 100 for c in CLASSES]
                     for x in runs[(model, norm, "random")]])
    return vals.mean(0)


def fig_results(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=True)
    y = np.arange(len(MODELS))[::-1]
    for ax, proto, title in zip(axes, ["random", "player"], ["Random split", "Unseen player"]):
        for yi, (row, (_, norm, label, color)) in zip(y, zip(rows, MODELS)):
            if row[proto] is None:
                continue
            m, s, _ = row[proto]
            alpha = 1.0 if norm or label == "Audio CRNN" else 0.45
            ax.barh(yi, m, height=0.62, color=color, alpha=alpha, edgecolor=SURFACE, linewidth=2)
            ax.errorbar(m, yi, xerr=s, color=INK2, capsize=0, lw=2)
            ax.text(m + s + 1.5, yi, f"{m:.1f}", va="center", color=INK, fontsize=24,
                    fontweight="bold" if label == "Fusion, normalised" else "normal")
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 50, 100])
        ax.set_title(title, loc="left", color=INK, fontsize=28, fontweight="bold", pad=14)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([m[2] for m in MODELS], color=INK)
    fig.supxlabel("macro-F1 (%), mean ± std: 3 seeds (random split), 3 folds (unseen player)", color=INK2, fontsize=22)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_per_class(runs, out):
    audio = per_class_f1(runs, "audio", "")
    fusion = per_class_f1(runs, "fusion", "_norm")
    order = np.argsort(audio)[::-1]
    fig, ax = plt.subplots(figsize=(13.6, 7.6))
    y = np.arange(len(CLASSES))
    for yi, i in zip(y, order):
        ax.plot([audio[i], fusion[i]], [yi, yi], color=RULE, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(audio[i], yi, s=260, color=AUDIO, zorder=2, edgecolor=SURFACE, linewidth=2)
        ax.scatter(fusion[i], yi, s=260, color=FUSION, zorder=3, edgecolor=SURFACE, linewidth=2)
        if fusion[i] - audio[i] >= 2:
            ax.text(min(fusion[i], audio[i]) - 1.2, yi, f"+{fusion[i] - audio[i]:.1f}",
                    ha="right", va="center", color=INK, fontsize=22, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([CLASSES[i] for i in order], color=INK)
    ax.invert_yaxis()
    lo = np.floor(min(audio.min(), fusion.min()) / 5) * 5 - 8
    ax.set_xlim(lo, 101)
    ax.set_xlabel("per-class F1 (%), mean over 3 seeds")
    ax.spines["left"].set_visible(False)
    ax.scatter([], [], s=260, color=AUDIO, label="Audio only")
    ax.scatter([], [], s=260, color=FUSION, label="Audio + hands")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2, fontsize=24, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    short = ["Dia", "Dua", "Dun", "Ji", "Pao", "Piz", "Por", "Tia", "Tre", "Tri", "Vib"]
    for ax, (model, norm, title) in zip(axes, [("audio", "", "Audio only"), ("fusion", "_norm", "Audio + hands")]):
        cm = sum(x["cm"] for x in runs[(model, norm, "random")])
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                if i != j and cm[i, j] > 0:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=21, color=INK)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(short, rotation=90, fontsize=17)
        ax.set_yticklabels(short, fontsize=17)
        ax.set_title(title, loc="left", color=INK, fontsize=26, fontweight="bold", pad=10)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("true", fontsize=20)
    for ax in axes:
        ax.set_xlabel("predicted (off-diagonal: error counts, 3 seeds)", fontsize=17)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def hand_frame(npz_path, hand_rank=0):
    lm = np.load(npz_path)["lm"]
    valid = ~np.isnan(lm[:, :, 0, 0])
    t = int(np.argmax(valid[:, hand_rank]))
    return lm[t, hand_rank, :, :2]


def normalise(h):
    rel = h - h[0]
    return rel / np.mean(np.linalg.norm(rel, axis=-1))


def fig_norm(npz_paths, labels, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
    for p, lab, color in zip(npz_paths, labels, PLAYERS):
        h = hand_frame(p)
        for ax, pts in zip(axes, [h, normalise(h)]):
            for a, b in HAND_EDGES:
                ax.plot(pts[[a, b], 0], pts[[a, b], 1], color=color, lw=3, solid_capstyle="round")
            ax.scatter(pts[:, 0], pts[:, 1], s=36, color=color, zorder=3)
        axes[1].plot([], [], color=color, lw=4, label=lab)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1, 0)
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].invert_yaxis()
    axes[0].set_title("Raw image coordinates", loc="left", color=INK, fontsize=26, fontweight="bold")
    axes[1].set_title("Wrist-centred, scale-normalised", loc="left", color=INK, fontsize=26, fontweight="bold")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(RULE)
            s.set_visible(True)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               fontsize=22, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, default=Path("lbd/poster/fig"))
    ap.add_argument("--font-dir", type=Path, required=True)
    ap.add_argument("--norm-npz", type=Path, nargs="*", default=[])
    ap.add_argument("--norm-labels", nargs="*", default=["Player 1", "Player 2", "Player 3"])
    args = ap.parse_args()

    setup_style(args.font_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs)
    rows = summary(runs)
    (args.out / "summary.json").write_text(json.dumps({
        "macro_f1": rows,
        "per_class_audio": dict(zip(CLASSES, per_class_f1(runs, "audio", "").round(1).tolist())),
        "per_class_fusion_norm": dict(zip(CLASSES, per_class_f1(runs, "fusion", "_norm").round(1).tolist())),
    }, indent=2))
    fig_results(rows, args.out / "results.pdf")
    fig_per_class(runs, args.out / "per_class.pdf")
    fig_confusion(runs, args.out / "confusion.pdf")
    if args.norm_npz:
        fig_norm(args.norm_npz, args.norm_labels, args.out / "norm.pdf")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
