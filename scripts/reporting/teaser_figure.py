from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

from src.augment import EvalTransform
from src.config import load_config
from src.dataset import build_dataloaders
from src.metrics import tta_forward
from src.model import build_model
from src.utils import load_checkpoint, seed_everything

CKPT = "results/phase2_bit_only/seed42/best_ema.pth"
CONFIG = "configs/phase2_main.yaml"
DATASET_JSON = "dataset/dataset.json"
DATASET_ROOT = Path("dataset")
OUT_PATH = Path("reports/figs/teaser.png")
TTA_OPS = ("orig", "hflip", "vflip", "rot180")

FAMILIES = ("object", "event", "attribute")
FAMILY_PROB_KEY = {"object": "probs_obj", "event": "probs_evt",
                   "attribute": "probs_attr"}
FAMILY_Y_KEY = {"object": "y_obj", "event": "y_evt", "attribute": "y_attr"}
FAMILY_TR = {"object": "NESNE", "event": "OLAY", "attribute": "ÖZNİTELİK"}

# Optional override (set to sample IDs to pin specific picks).
PIN_SUCCESS_ID: str | None = None
PIN_FAILURE_ID: str | None = None

# Color palette: fill / edge / text
PAL = {
    "tp": ("#d7f3d7", "#2e7d32", "#16611b"),
    "fn": ("#ffe0e0", "#c62828", "#9c1c1c"),
    "fp": ("#ffe8cc", "#ef6c00", "#b35100"),
}
ROW_COLOR = {"success": "#2e7d32", "failure": "#c62828"}
ROW_TITLE = {"success": "BAŞARI", "failure": "HATA"}
GRAY = "#6b6b6b"
PRIMARY = "#1a1a1a"
DIVIDER = "#e2e2e2"
PANEL_BG = "#fafafa"
THRESHOLD = 0.5


def _comma(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _per_sample_macro_f1(probs: np.ndarray, targets: np.ndarray,
                         threshold: float = THRESHOLD) -> np.ndarray:
    preds = (probs >= threshold).astype(np.int64)
    y = targets.astype(np.int64)
    tp = ((preds == 1) & (y == 1)).sum(axis=1)
    fp = ((preds == 1) & (y == 0)).sum(axis=1)
    fn = ((preds == 0) & (y == 1)).sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.astype(np.float32)


def _chip_kind(prob: float, in_gt: bool) -> str | None:
    """TP / FN / FP / (TN -> None, not shown)."""
    pred = prob >= THRESHOLD
    if in_gt and pred:
        return "tp"
    if in_gt and not pred:
        return "fn"
    if (not in_gt) and pred:
        return "fp"
    return None


def _chip_text(kind: str, label: str, prob: float) -> str:
    if kind == "tp":
        return f"✓  {label}  {_comma(prob)}"
    if kind == "fn":
        return f"✗  {label}"
    if kind == "fp":
        return f"✗  {label}  {_comma(prob)}"
    return label


def _gather_chips(probs: np.ndarray, targets: np.ndarray,
                  vocab: list[str]) -> list[tuple[str, str, float]]:
    """List of (kind, label, prob) chips for a single sample in one family.
    Order: TP by p desc, then FN alpha, then FP by p desc."""
    out: list[tuple[str, str, float]] = []
    for i, name in enumerate(vocab):
        kind = _chip_kind(float(probs[i]), bool(targets[i] > 0.5))
        if kind is None:
            continue
        out.append((kind, name, float(probs[i])))
    rank = {"tp": 0, "fn": 1, "fp": 2}
    out.sort(key=lambda c: (rank[c[0]], -c[2] if c[0] != "fn" else 0, c[1]))
    return out


def _square_center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _place_chip(ax, text: str, kind: str, *, x: float, y: float, fontsize: float):
    """Draw a chip at (x, y) in axes coordinates, returning the text artist
    so the caller can measure it after a draw."""
    fill, edge, fg = PAL[kind]
    return ax.text(
        x, y, text,
        transform=ax.transAxes, fontsize=fontsize,
        color=fg, weight="bold", va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor=fill, edgecolor=edge, linewidth=0.7),
    )


def _ax_width_of(ax, text_artist) -> float:
    """Width of a placed text artist, expressed in axes (transAxes) units."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    bb = text_artist.get_window_extent(renderer)
    inv = ax.transAxes.inverted()
    p0 = inv.transform((bb.x0, bb.y0))
    p1 = inv.transform((bb.x1, bb.y1))
    return float(p1[0] - p0[0])


def _flow_chips(ax, chips: list[tuple[str, str, float]], *,
                x0: float, y_top: float, max_x: float,
                line_h: float, fontsize: float, gap: float = 0.012) -> float:
    """Place chips left-to-right within [x0, max_x], wrapping to a new line
    when needed. Returns the y of the last line (axes coords)."""
    if not chips:
        ax.text(x0, y_top, "(yok)", transform=ax.transAxes,
                fontsize=fontsize - 0.5, color=GRAY, style="italic",
                va="center", ha="left")
        return y_top
    # Force a draw so text extents are measurable.
    ax.figure.canvas.draw()
    x, y = x0, y_top
    for kind, label, prob in chips:
        artist = _place_chip(ax, _chip_text(kind, label, prob), kind,
                             x=x, y=y, fontsize=fontsize)
        w = _ax_width_of(ax, artist)
        if x + w > max_x and x > x0:
            artist.remove()
            x = x0
            y -= line_h
            artist = _place_chip(ax, _chip_text(kind, label, prob), kind,
                                 x=x, y=y, fontsize=fontsize)
            w = _ax_width_of(ax, artist)
        x += w + gap
    return y


def _draw_status_strip(fig, rect_inches: tuple[float, float, float, float],
                       fig_w: float, fig_h: float,
                       row_kind: str) -> None:
    x, y, w, h = rect_inches
    ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    color = ROW_COLOR[row_kind]
    patch = FancyBboxPatch(
        (0.05, 0.03), 0.9, 0.94,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        facecolor=color, edgecolor=color, linewidth=0,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(0.5, 0.5, ROW_TITLE[row_kind],
            transform=ax.transAxes, rotation=90,
            ha="center", va="center", color="white",
            fontsize=12, weight="bold")


def _draw_image_panel(fig, img: Image.Image,
                      rect_inches: tuple[float, float, float, float],
                      fig_w: float, fig_h: float, edge_color: str,
                      title: str | None) -> None:
    x, y, w, h = rect_inches
    ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
    img_sq = _square_center_crop(img)
    ax.imshow(img_sq)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge_color)
        s.set_linewidth(1.6)
    if title:
        ax.set_title(title, fontsize=9.5, weight="bold",
                     color=PRIMARY, pad=4)


def _draw_panel(fig, rect_inches: tuple[float, float, float, float],
                fig_w: float, fig_h: float, *,
                row_kind: str, f1: float, p_change: float,
                chips_per_family: dict[str, list],
                column_title: str | None) -> None:
    x, y, w, h = rect_inches
    ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # subtle background
    bg = Rectangle((0, 0), 1, 1, facecolor=PANEL_BG,
                   edgecolor="#ececec", linewidth=0.6,
                   transform=ax.transAxes)
    ax.add_patch(bg)

    if column_title:
        ax.set_title(column_title, fontsize=9.5, weight="bold",
                     color=PRIMARY, pad=4, loc="left", x=0.02)

    # Header line: F1 (left, row color) + P(değişim) (right, gray)
    ax.text(0.025, 0.92, f"F1 = {_comma(f1)}",
            transform=ax.transAxes, fontsize=11, weight="bold",
            color=ROW_COLOR[row_kind], va="center", ha="left")
    ax.text(0.975, 0.92, f"P(değişim) = {_comma(p_change)}",
            transform=ax.transAxes, fontsize=10,
            color=GRAY, va="center", ha="right")

    # Divider under header
    ax.plot([0.02, 0.98], [0.83, 0.83], color=DIVIDER, linewidth=0.7,
            transform=ax.transAxes)

    # Per family block
    n_fam = len(FAMILIES)
    block_top = 0.78
    block_bottom = 0.06
    block_height = (block_top - block_bottom) / n_fam
    label_w = 0.23  # space reserved for family label ("ÖZNİTELİK" needs ~0.21)
    chip_x0 = label_w + 0.02
    chip_max_x = 0.97
    line_h = 0.085

    for i, fam in enumerate(FAMILIES):
        center_y = block_top - (i + 0.5) * block_height
        # Family label
        ax.text(0.025, center_y, FAMILY_TR[fam],
                transform=ax.transAxes, fontsize=8.5, weight="bold",
                color=GRAY, va="center", ha="left", family="sans-serif")
        chips = chips_per_family[fam]
        _flow_chips(
            ax, chips,
            x0=chip_x0,
            y_top=center_y + 0.5 * line_h * (max(0, _wrap_lines_estimate(chips)) - 1),
            max_x=chip_max_x,
            line_h=line_h,
            fontsize=7.5,
        )
        # Bottom divider between blocks (faint), skip last
        if i < n_fam - 1:
            div_y = block_top - (i + 1) * block_height
            ax.plot([0.02, 0.98], [div_y, div_y], color=DIVIDER,
                    linewidth=0.5, transform=ax.transAxes)


def _wrap_lines_estimate(chips: list[tuple[str, str, float]]) -> int:
    """Rough vertical centering hint -- assume ~5 chips per line."""
    if not chips:
        return 1
    return max(1, (len(chips) + 4) // 5)


def _draw_column_titles(fig, fig_w: float, fig_h: float, layout: dict) -> None:
    # A title
    cx_a = (layout["a_x"] + layout["img_size"] / 2) / fig_w
    # B title
    cx_b = (layout["b_x"] + layout["img_size"] / 2) / fig_w
    # Panel title (left-aligned)
    px = (layout["panel_x"] + 0.06) / fig_w
    y_t = (fig_h - 0.18) / fig_h
    fig.text(cx_a, y_t, "A  (öncesi)", ha="center", va="center",
             fontsize=10, weight="bold", color=PRIMARY)
    fig.text(cx_b, y_t, "B  (sonrası)", ha="center", va="center",
             fontsize=10, weight="bold", color=PRIMARY)
    fig.text(px, y_t, "Tahmin · gerçek  (eşik 0,5)", ha="left", va="center",
             fontsize=10, weight="bold", color=PRIMARY)


def _draw_legend(fig, fig_w: float, fig_h: float, y_in: float) -> None:
    items = [
        ("tp", "doğru (TP)", "✓"),
        ("fn", "kaçırılan (FN)", "✗"),
        ("fp", "yanlış pozitif (FP)", "✗"),
    ]
    # axes spanning bottom area
    ax = fig.add_axes([0.05, y_in / fig_h, 0.92, 0.32 / fig_h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    x = 0.0
    for kind, text, mark in items:
        artist = _place_chip(ax, f"{mark}  {text}", kind, x=x, y=0.5, fontsize=8)
        ax.figure.canvas.draw()
        w = _ax_width_of(ax, artist)
        x += w + 0.03


def render_card_figure(picks_data: list[dict], out_path: Path,
                       *, img_size: float = 1.95) -> None:
    """N-row card layout: per row a status strip, A/B images, and a panel
    with F1 + P(change) header and per-family TP/FN/FP chips.

    Each pick dict has: sample_id, f1, p_change, row_kind, a_img, b_img,
    chips_per_family.
    """
    n_rows = len(picks_data)
    if n_rows == 0:
        raise ValueError("picks_data must be non-empty")

    strip_w = 0.32
    gap_x = 0.10
    left_margin = 0.18
    panel_left = left_margin + strip_w + gap_x + 2 * img_size + 2 * gap_x
    row_gap = 0.18
    top_pad_for_titles = 0.45
    legend_h = 0.55

    fig_w = 8.4
    fig_h = top_pad_for_titles + n_rows * img_size + (n_rows - 1) * row_gap + legend_h
    panel_w = fig_w - panel_left - 0.18

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    layout = {
        "img_size": img_size,
        "a_x": left_margin + strip_w + gap_x,
        "b_x": left_margin + strip_w + gap_x + img_size + gap_x,
        "panel_x": panel_left,
    }

    rows_top = fig_h - top_pad_for_titles
    row_ys = [rows_top - i * (img_size + row_gap) - img_size for i in range(n_rows)]

    for pk, y in zip(picks_data, row_ys):
        edge = ROW_COLOR[pk["row_kind"]]
        _draw_status_strip(
            fig, (left_margin, y, strip_w, img_size), fig_w, fig_h, pk["row_kind"]
        )
        _draw_image_panel(
            fig, pk["a_img"], (layout["a_x"], y, img_size, img_size),
            fig_w, fig_h, edge, title=None,
        )
        _draw_image_panel(
            fig, pk["b_img"], (layout["b_x"], y, img_size, img_size),
            fig_w, fig_h, edge, title=None,
        )
        _draw_panel(
            fig,
            (layout["panel_x"], y - 0.05, panel_w, img_size + 0.10),
            fig_w, fig_h,
            row_kind=pk["row_kind"], f1=pk["f1"], p_change=pk["p_change"],
            chips_per_family=pk["chips_per_family"],
            column_title=None,
        )

    _draw_column_titles(fig, fig_w, fig_h, layout)

    legend_y = max(0.15, row_ys[-1] - legend_h - 0.05)
    _draw_legend(fig, fig_w, fig_h, legend_y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"saved -> {out_path}")


def render_teaser(picks_data: list[dict]) -> None:
    render_card_figure(picks_data, OUT_PATH)


def main() -> None:
    cfg = load_config(CONFIG)
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = json.loads(Path(cfg["data"]["vocab_path"]).read_text())

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(CKPT)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = model.encoder.norm_stats()
    transform = EvalTransform(img_size=int(cfg["data"].get("img_size", 224)),
                              mean=mean, std=std)
    _, _, test_loader = build_dataloaders(
        cfg, transform_train=transform, transform_eval=transform
    )

    all_probs: dict[str, list[np.ndarray]] = {fam: [] for fam in FAMILIES}
    all_targets: dict[str, list[np.ndarray]] = {fam: [] for fam in FAMILIES}
    all_change: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    for batch in test_loader:
        a = batch["A"].to(device, non_blocking=True)
        b = batch["B"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = tta_forward(model, {"A": a, "B": b}, list(TTA_OPS))
        all_change.append(out["prob_nochg"].cpu().numpy())
        for fam in FAMILIES:
            all_probs[fam].append(out[FAMILY_PROB_KEY[fam]].cpu().numpy())
            all_targets[fam].append(batch[FAMILY_Y_KEY[fam]].cpu().numpy())
        all_sample_ids.extend(batch["sample_id"])

    change_probs = np.concatenate(all_change, axis=0)
    fam_probs = {fam: np.concatenate(all_probs[fam], axis=0) for fam in FAMILIES}
    fam_targets = {fam: np.concatenate(all_targets[fam], axis=0) for fam in FAMILIES}
    for fam in FAMILIES:
        fam_probs[fam] = fam_probs[fam] * change_probs[:, None]

    per_fam_f1 = {fam: _per_sample_macro_f1(fam_probs[fam], fam_targets[fam])
                  for fam in FAMILIES}
    has_pos = np.zeros(len(all_sample_ids), dtype=bool)
    for fam in FAMILIES:
        has_pos |= fam_targets[fam].sum(axis=1) > 0
    sample_f1 = np.mean([per_fam_f1[fam] for fam in FAMILIES], axis=0)

    candidates = np.where(has_pos)[0]
    if PIN_SUCCESS_ID is not None:
        top_succ = all_sample_ids.index(PIN_SUCCESS_ID)
    else:
        top_succ = int(candidates[np.argmax(sample_f1[candidates])])
    if PIN_FAILURE_ID is not None:
        bot_fail = all_sample_ids.index(PIN_FAILURE_ID)
    else:
        bot_fail = int(candidates[np.argmin(sample_f1[candidates])])

    blob = json.loads(Path(DATASET_JSON).read_text())
    by_id = {r["sample_id"]: r for r in blob["images"]}

    picks_data: list[dict] = []
    for idx, row_kind in [(top_succ, "success"), (bot_fail, "failure")]:
        rec = by_id[all_sample_ids[idx]]
        a_img = Image.open(DATASET_ROOT / rec["rgb_A"]).convert("RGB")
        b_img = Image.open(DATASET_ROOT / rec["rgb_B"]).convert("RGB")
        chips_per_family = {}
        for fam in FAMILIES:
            chips_per_family[fam] = _gather_chips(
                fam_probs[fam][idx], fam_targets[fam][idx], vocab[fam]
            )
        picks_data.append({
            "sample_id": all_sample_ids[idx],
            "f1": float(sample_f1[idx]),
            "p_change": float(change_probs[idx]),
            "row_kind": row_kind,
            "a_img": a_img,
            "b_img": b_img,
            "chips_per_family": chips_per_family,
        })
        print(f"[{row_kind}] {all_sample_ids[idx]}  "
              f"F1={sample_f1[idx]:.3f}  P_chg={change_probs[idx]:.3f}")

    render_teaser(picks_data)


if __name__ == "__main__":
    main()
