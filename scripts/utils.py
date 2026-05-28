from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REGULAR_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
BOLD_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyhbd.ttc",
    *REGULAR_FONT_CANDIDATES,
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # 图表里有中文标题，优先尝试系统中文字体，避免在不同机器上显示成方块。
    candidates = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int) -> None:
    draw.text((52, 28), title, fill=(25, 32, 45), font=load_font(34, bold=True))
    draw.line((52, 82, width - 52, 82), fill=(211, 218, 229), width=2)


def save_bar_chart(
    data: list[tuple[str, float]],
    title: str,
    path: Path,
    *,
    unit: str = "",
    width: int = 1280,
    height: int = 760,
    color: tuple[int, int, int] = (38, 116, 161),
    left_margin: int = 190,
    right_margin: int = 70,
    label_max_chars: int = 16,
    value_format: str = "{value:,.0f}{unit}",
    label_font_size: int = 21,
    value_font_size: int = 18,
    min_bar_height: int = 22,
    gap: int = 12,
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width)

    label_font = load_font(label_font_size)
    value_font = load_font(value_font_size)
    left, top, right, bottom = left_margin, 125, width - right_margin, height - 95
    max_v = max([value for _, value in data] or [1])
    n = max(len(data), 1)
    bar_h = max(min_bar_height, int((bottom - top - gap * (n - 1)) / n))

    for i, (label, value) in enumerate(data):
        y = top + i * (bar_h + gap)
        bar_w = int((right - left) * value / max_v) if max_v else 0
        draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=5, fill=color)
        draw.text((52, y + 2), label[:label_max_chars], fill=(45, 55, 72), font=label_font)
        value_text = value_format.format(value=value, unit=unit)
        draw.text((left + bar_w + 10, y + 1), value_text, fill=(45, 55, 72), font=value_font)

    img.save(path)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # 35 已经足够让概率接近 0 或 1，同时可以避免 exp 溢出。
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (prob >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
