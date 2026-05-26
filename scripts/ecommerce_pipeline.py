from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ecommerce"
PROCESSED_DIR = ROOT / "data" / "processed" / "ecommerce"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"

FILES = [
    RAW_DIR / "2019-Oct.csv.gz",
    RAW_DIR / "2019-Nov.csv.zip",
    RAW_DIR / "2019-Nov.csv.gz",
    RAW_DIR / "2019-Dec.csv.gz",
]

CHUNKSIZE = 1_000_000
USER_SAMPLE_MOD = 40
RANDOM_SEED = 2026

USECOLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

PRICE_BINS = [0, 10, 50, 100, 200, 500, 1000, 5000, np.inf]
PRICE_LABELS = [
    "0-10",
    "10-50",
    "50-100",
    "100-200",
    "200-500",
    "500-1000",
    "1000-5000",
    "5000+",
]


def ensure_dirs() -> None:
    for directory in [PROCESSED_DIR, FIG_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def existing_input_files() -> list[Path]:
    files: list[Path] = []
    seen_names: set[str] = set()
    for file_path in FILES:
        if not file_path.exists():
            continue
        logical_name = file_path.name.replace(".zip", "").replace(".gz", "")
        if logical_name in seen_names:
            continue
        seen_names.add(logical_name)
        files.append(file_path)
    return files


def add_metrics(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["event_time"] = pd.to_datetime(chunk["event_time"], utc=True, errors="coerce")
    chunk = chunk.dropna(subset=["event_time"])
    chunk["date"] = chunk["event_time"].dt.strftime("%Y-%m-%d")
    chunk["hour"] = chunk["event_time"].dt.hour.astype("int16")
    chunk["month"] = chunk["date"].str.slice(0, 7)
    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").fillna(0).clip(lower=0)
    chunk["brand"] = chunk["brand"].fillna("unknown").astype(str)
    chunk["category_code"] = chunk["category_code"].fillna("unknown").astype(str)
    chunk["user_session"] = chunk["user_session"].fillna("unknown").astype(str)
    chunk["is_view"] = (chunk["event_type"] == "view").astype("int8")
    chunk["is_cart"] = (chunk["event_type"] == "cart").astype("int8")
    chunk["is_purchase"] = (chunk["event_type"] == "purchase").astype("int8")
    chunk["revenue"] = np.where(chunk["event_type"] == "purchase", chunk["price"], 0.0)
    chunk["price_bin"] = pd.cut(
        chunk["price"],
        bins=PRICE_BINS,
        labels=PRICE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)
    return chunk


def concat_group(parts: list[pd.DataFrame], keys: list[str], out_name: str) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    agg: dict[str, str] = {}
    for column in df.columns:
        if column in keys:
            continue
        agg[column] = "mean" if column.startswith("avg_") else "sum"
    result = df.groupby(keys, as_index=False).agg(agg)
    result.to_csv(PROCESSED_DIR / out_name, index=False, encoding="utf-8-sig")
    return result


def aggregate_raw_data() -> dict[str, pd.DataFrame | dict]:
    files = existing_input_files()
    if not files:
        raise FileNotFoundError(f"No ecommerce input files found in {RAW_DIR}")

    event_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    month_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    category_parts: list[pd.DataFrame] = []
    brand_parts: list[pd.DataFrame] = []
    user_parts: list[pd.DataFrame] = []

    sampled_stage_users = {"view": set(), "cart": set(), "purchase": set()}
    sampled_month_stage_users: dict[tuple[str, str], set] = {}

    total_rows = 0
    t0 = time.time()
    print("Input files:")
    for file_path in files:
        print(f"  {file_path.name}: {file_path.stat().st_size / 1024 ** 3:.2f} GB")

    for file_path in files:
        print(f"\nProcessing {file_path.name}")
        reader = pd.read_csv(
            file_path,
            usecols=USECOLS,
            chunksize=CHUNKSIZE,
            low_memory=False,
            compression="infer",
        )

        for chunk_id, chunk in enumerate(reader, start=1):
            total_rows += len(chunk)
            elapsed = (time.time() - t0) / 60
            print(
                f"  chunk {chunk_id:03d}, rows={len(chunk):,}, "
                f"total={total_rows:,}, elapsed={elapsed:.1f} min",
                flush=True,
            )

            chunk = add_metrics(chunk)

            event_parts.append(
                chunk.groupby("event_type", as_index=False).agg(
                    events=("event_type", "size"),
                    users_chunk_sum=("user_id", "nunique"),
                    sessions_chunk_sum=("user_session", "nunique"),
                    revenue=("revenue", "sum"),
                    price_sum=("price", "sum"),
                )
            )

            daily_parts.append(
                chunk.groupby(["date", "event_type"], as_index=False).agg(
                    events=("event_type", "size"),
                    users_chunk_sum=("user_id", "nunique"),
                    revenue=("revenue", "sum"),
                )
            )

            hourly_parts.append(
                chunk.groupby(["hour", "event_type"], as_index=False).agg(
                    events=("event_type", "size"),
                    users_chunk_sum=("user_id", "nunique"),
                    revenue=("revenue", "sum"),
                )
            )

            month_parts.append(
                chunk.groupby(["month", "event_type"], as_index=False).agg(
                    events=("event_type", "size"),
                    users_chunk_sum=("user_id", "nunique"),
                    revenue=("revenue", "sum"),
                )
            )

            price_parts.append(
                chunk.groupby(["price_bin", "event_type"], observed=False, as_index=False).agg(
                    events=("event_type", "size"),
                    users_chunk_sum=("user_id", "nunique"),
                    revenue=("revenue", "sum"),
                )
            )

            category_parts.append(
                chunk.groupby("category_code", as_index=False).agg(
                    events=("event_type", "size"),
                    views=("is_view", "sum"),
                    carts=("is_cart", "sum"),
                    purchases=("is_purchase", "sum"),
                    users_chunk_sum=("user_id", "nunique"),
                    revenue=("revenue", "sum"),
                    avg_price=("price", "mean"),
                )
            )

            brand_parts.append(
                chunk.groupby("brand", as_index=False).agg(
                    events=("event_type", "size"),
                    views=("is_view", "sum"),
                    carts=("is_cart", "sum"),
                    purchases=("is_purchase", "sum"),
                    users_chunk_sum=("user_id", "nunique"),
                    revenue=("revenue", "sum"),
                    avg_price=("price", "mean"),
                )
            )

            sampled = chunk.loc[(chunk["user_id"].astype("int64") % USER_SAMPLE_MOD) == 0].copy()
            if len(sampled) > 0:
                for event_name in ["view", "cart", "purchase"]:
                    ids = sampled.loc[sampled["event_type"] == event_name, "user_id"].unique()
                    sampled_stage_users[event_name].update(ids.tolist())
                    for month_value, month_ids in sampled.loc[
                        sampled["event_type"] == event_name, ["month", "user_id"]
                    ].groupby("month")["user_id"]:
                        sampled_month_stage_users.setdefault((month_value, event_name), set()).update(
                            month_ids.unique().tolist()
                        )

                sampled["price_event_count"] = 1
                sampled["price_sum"] = sampled["price"]
                user_parts.append(
                    sampled.groupby(["month", "user_id"], as_index=False).agg(
                        total_events=("event_type", "size"),
                        views=("is_view", "sum"),
                        carts=("is_cart", "sum"),
                        purchases=("is_purchase", "sum"),
                        unique_products=("product_id", "nunique"),
                        unique_categories=("category_id", "nunique"),
                        unique_brands=("brand", "nunique"),
                        sessions=("user_session", "nunique"),
                        price_sum=("price_sum", "sum"),
                        price_event_count=("price_event_count", "sum"),
                        max_price=("price", "max"),
                        active_days=("date", "nunique"),
                    )
                )

            del chunk

    event_summary = concat_group(event_parts, ["event_type"], "event_type_summary.csv")
    daily_summary = concat_group(daily_parts, ["date", "event_type"], "daily_event_summary.csv")
    hourly_summary = concat_group(hourly_parts, ["hour", "event_type"], "hourly_event_summary.csv")
    month_summary = concat_group(month_parts, ["month", "event_type"], "month_event_summary.csv")
    price_summary = concat_group(price_parts, ["price_bin", "event_type"], "price_bin_summary.csv")
    category_summary = concat_group(category_parts, ["category_code"], "category_summary.csv")
    brand_summary = concat_group(brand_parts, ["brand"], "brand_summary.csv")

    category_summary = category_summary.sort_values(["purchases", "revenue"], ascending=False)
    brand_summary = brand_summary.sort_values(["purchases", "revenue"], ascending=False)
    category_summary.head(60).to_csv(PROCESSED_DIR / "top_categories.csv", index=False, encoding="utf-8-sig")
    brand_summary.head(60).to_csv(PROCESSED_DIR / "top_brands.csv", index=False, encoding="utf-8-sig")

    user_features = pd.concat(user_parts, ignore_index=True)
    user_features = user_features.groupby(["month", "user_id"], as_index=False).agg(
        total_events=("total_events", "sum"),
        views=("views", "sum"),
        carts=("carts", "sum"),
        purchases=("purchases", "sum"),
        unique_products=("unique_products", "sum"),
        unique_categories=("unique_categories", "sum"),
        unique_brands=("unique_brands", "sum"),
        sessions=("sessions", "sum"),
        price_sum=("price_sum", "sum"),
        price_event_count=("price_event_count", "sum"),
        max_price=("max_price", "max"),
        active_days=("active_days", "sum"),
    )
    user_features["avg_price"] = user_features["price_sum"] / user_features["price_event_count"].clip(lower=1)
    user_features["converted"] = (user_features["purchases"] > 0).astype("int8")
    user_features.to_csv(PROCESSED_DIR / "user_features_sample.csv", index=False, encoding="utf-8-sig")

    funnel = pd.DataFrame(
        [
            {"stage": stage, "sample_users": len(users), "estimated_users": len(users) * USER_SAMPLE_MOD}
            for stage, users in sampled_stage_users.items()
        ]
    )
    base = max(float(funnel.loc[funnel["stage"] == "view", "estimated_users"].iloc[0]), 1.0)
    funnel["relative_to_view"] = funnel["estimated_users"] / base
    funnel.to_csv(PROCESSED_DIR / "funnel_summary.csv", index=False, encoding="utf-8-sig")

    month_funnel_rows = []
    for (month_value, stage), users in sampled_month_stage_users.items():
        month_funnel_rows.append(
            {
                "month": month_value,
                "stage": stage,
                "sample_users": len(users),
                "estimated_users": len(users) * USER_SAMPLE_MOD,
            }
        )
    month_funnel = pd.DataFrame(month_funnel_rows)
    month_funnel.to_csv(PROCESSED_DIR / "month_funnel_summary.csv", index=False, encoding="utf-8-sig")

    meta = {
        "total_rows": int(total_rows),
        "input_files": [p.name for p in files],
        "user_sample_mod": USER_SAMPLE_MOD,
        "chunk_size": CHUNKSIZE,
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
        "note": "Unique user counts in funnel tables are estimated by deterministic user_id modulo sampling.",
    }
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "meta": meta,
        "event_summary": event_summary,
        "daily_summary": daily_summary,
        "hourly_summary": hourly_summary,
        "month_summary": month_summary,
        "price_summary": price_summary,
        "category_summary": category_summary,
        "brand_summary": brand_summary,
        "funnel": funnel,
        "month_funnel": month_funnel,
        "user_features": user_features,
    }


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
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
    unit: str = "",
    width: int = 1280,
    height: int = 760,
    color: tuple[int, int, int] = (38, 116, 161),
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(34, bold=True)
    label_font = load_font(21)
    small_font = load_font(18)
    draw.text((52, 28), title, fill=(25, 32, 45), font=title_font)
    draw.line((52, 82, width - 52, 82), fill=(211, 218, 229), width=2)

    left, top, right, bottom = 190, 125, width - 70, height - 95
    values = [v for _, v in data]
    max_v = max(values) if values else 1
    n = max(len(data), 1)
    gap = 12
    bar_h = max(22, int((bottom - top - gap * (n - 1)) / n))

    for i, (label, value) in enumerate(data):
        y = top + i * (bar_h + gap)
        bar_w = int((right - left) * (value / max_v)) if max_v else 0
        draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=5, fill=color)
        draw.text((52, y + 2), label[:16], fill=(45, 55, 72), font=label_font)
        value_text = f"{value:,.0f}{unit}"
        draw.text((left + bar_w + 10, y + 1), value_text, fill=(45, 55, 72), font=small_font)

    img.save(path)


def save_line_chart(
    series: dict[str, list[tuple[str, float]]],
    title: str,
    path: Path,
    width: int = 1400,
    height: int = 820,
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(34, bold=True)
    label_font = load_font(19)
    draw.text((52, 28), title, fill=(25, 32, 45), font=title_font)
    draw.line((52, 82, width - 52, 82), fill=(211, 218, 229), width=2)

    left, top, right, bottom = 90, 125, width - 80, height - 115
    all_dates = sorted({date for points in series.values() for date, _ in points})
    value_max = max([value for points in series.values() for _, value in points] or [1])
    value_max = value_max * 1.08

    draw.line((left, bottom, right, bottom), fill=(95, 108, 128), width=2)
    draw.line((left, top, left, bottom), fill=(95, 108, 128), width=2)
    for i in range(5):
        y = bottom - (bottom - top) * i / 4
        draw.line((left, y, right, y), fill=(234, 238, 245), width=1)
        draw.text((20, y - 10), f"{value_max * i / 4 / 1_000_000:.1f}M", fill=(75, 85, 99), font=label_font)

    colors = {
        "view": (48, 112, 173),
        "cart": (218, 135, 42),
        "purchase": (47, 139, 85),
    }
    for name, points in series.items():
        point_map = dict(points)
        coords = []
        for idx, date in enumerate(all_dates):
            x = left + (right - left) * idx / max(len(all_dates) - 1, 1)
            y = bottom - (bottom - top) * point_map.get(date, 0) / value_max
            coords.append((x, y))
        if len(coords) > 1:
            draw.line(coords, fill=colors.get(name, (60, 60, 60)), width=4)
        for x, y in coords[:: max(1, len(coords) // 12)]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors.get(name, (60, 60, 60)))

    legend_x = left
    for name in series:
        color = colors.get(name, (60, 60, 60))
        draw.rectangle((legend_x, height - 68, legend_x + 24, height - 44), fill=color)
        draw.text((legend_x + 34, height - 70), name, fill=(45, 55, 72), font=label_font)
        legend_x += 150

    for idx in np.linspace(0, len(all_dates) - 1, min(7, len(all_dates)), dtype=int):
        x = left + (right - left) * idx / max(len(all_dates) - 1, 1)
        draw.text((x - 45, bottom + 18), all_dates[idx][5:], fill=(75, 85, 99), font=label_font)

    img.save(path)


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    labels = [["TN", "FP"], ["FN", "TP"]]
    img = Image.new("RGB", (900, 650), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, "购买预测模型混淆矩阵", 900)
    big_font = load_font(34, bold=True)
    label_font = load_font(23)
    colors = [[(225, 238, 248), (252, 235, 208)], [(252, 235, 208), (220, 241, 226)]]
    x0, y0, cell = 225, 145, 210
    for r in range(2):
        for c in range(2):
            x, y = x0 + c * cell, y0 + r * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=colors[r][c], outline=(130, 145, 166), width=2)
            draw.text((x + 82, y + 42), labels[r][c], fill=(35, 45, 60), font=label_font)
            draw.text((x + 72, y + 102), f"{int(cm[r, c]):,}", fill=(20, 30, 45), font=big_font)
    draw.text((x0 + 65, y0 - 45), "Pred 0", fill=(55, 65, 80), font=label_font)
    draw.text((x0 + cell + 65, y0 - 45), "Pred 1", fill=(55, 65, 80), font=label_font)
    draw.text((90, y0 + 80), "True 0", fill=(55, 65, 80), font=label_font)
    draw.text((90, y0 + cell + 80), "True 1", fill=(55, 65, 80), font=label_font)
    img.save(path)


def build_charts(results: dict[str, pd.DataFrame | dict], model_result: dict) -> None:
    event_summary = results["event_summary"].copy()
    order = ["view", "cart", "purchase"]
    event_summary["event_type"] = pd.Categorical(event_summary["event_type"], order, ordered=True)
    event_summary = event_summary.sort_values("event_type")
    save_bar_chart(
        [(row.event_type, row.events) for row in event_summary.itertuples()],
        "三个月用户行为事件量",
        FIG_DIR / "event_type_summary.png",
        color=(42, 111, 151),
    )

    funnel = results["funnel"].copy()
    funnel["stage"] = pd.Categorical(funnel["stage"], order, ordered=True)
    funnel = funnel.sort_values("stage")
    save_bar_chart(
        [(row.stage, row.estimated_users) for row in funnel.itertuples()],
        "转化漏斗估计用户规模",
        FIG_DIR / "funnel_users.png",
        color=(55, 139, 97),
    )

    daily = results["daily_summary"].copy()
    line_series = {}
    for event_type in order:
        subset = daily.loc[daily["event_type"] == event_type].sort_values("date")
        line_series[event_type] = [(row.date, row.events) for row in subset.itertuples()]
    save_line_chart(line_series, "三个月日度行为变化", FIG_DIR / "daily_events.png")

    monthly = results["month_summary"].copy()
    monthly_pivot = monthly.pivot_table(index="month", columns="event_type", values="events", aggfunc="sum").fillna(0)
    month_data = []
    for month in monthly_pivot.index:
        purchases = float(monthly_pivot.loc[month].get("purchase", 0))
        carts = float(monthly_pivot.loc[month].get("cart", 0))
        month_data.append((month, purchases / max(carts, 1) * 100))
    save_bar_chart(
        month_data,
        "各月加购到购买转化率",
        FIG_DIR / "monthly_cart_purchase_rate.png",
        unit="%",
        color=(213, 128, 55),
    )

    top_categories = results["category_summary"].copy().sort_values("revenue", ascending=False).head(12)
    save_bar_chart(
        [(row.category_code if row.category_code != "unknown" else "unknown", row.revenue) for row in top_categories.itertuples()],
        "销售额最高的品类 Top 12",
        FIG_DIR / "top_categories_revenue.png",
        color=(84, 107, 171),
    )

    top_brands = results["brand_summary"].copy().sort_values("revenue", ascending=False).head(12)
    save_bar_chart(
        [(row.brand if row.brand != "unknown" else "unknown", row.revenue) for row in top_brands.itertuples()],
        "销售额最高的品牌 Top 12",
        FIG_DIR / "top_brands_revenue.png",
        color=(132, 93, 154),
    )

    feature_importance = model_result["feature_importance"]
    save_bar_chart(
        [(item["feature"], abs(item["coef"])) for item in feature_importance[:10]],
        "购买预测模型特征重要性",
        FIG_DIR / "model_feature_importance.png",
        color=(37, 132, 132),
    )
    save_confusion_matrix(np.array(model_result["confusion_matrix"]), FIG_DIR / "model_confusion_matrix.png")


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_model(user_features: pd.DataFrame) -> dict:
    feature_cols = [
        "total_events",
        "views",
        "carts",
        "unique_products",
        "unique_categories",
        "unique_brands",
        "sessions",
        "avg_price",
        "max_price",
        "active_days",
    ]
    df = user_features.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["converted"])
    if len(df) > 250_000:
        sampled_parts = []
        for _, part in df.groupby("converted"):
            take = max(1, int(250_000 * len(part) / len(df)))
            sampled_parts.append(part.sample(min(take, len(part)), random_state=RANDOM_SEED))
        df = pd.concat(sampled_parts, ignore_index=True)

    x = np.log1p(df[feature_cols].astype(float).to_numpy())
    y = df["converted"].astype(int).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(df))
    split = int(len(df) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]

    pos_rate = y_train.mean()
    pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
    weights = np.where(y_train == 1, min(pos_weight, 20.0), 1.0)

    beta = np.zeros(x_train.shape[1])
    lr = 0.08
    reg = 0.002
    for _ in range(700):
        pred = sigmoid(x_train @ beta)
        error = (pred - y_train) * weights
        grad = (x_train.T @ error) / weights.sum()
        grad[1:] += reg * beta[1:]
        beta -= lr * grad

    prob = sigmoid(x_test @ beta)
    precision_recall = []
    for threshold in np.linspace(0.15, 0.75, 25):
        y_pred = (prob >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        precision_recall.append((f1, threshold, precision, recall))
    _, threshold, _, _ = max(precision_recall)
    y_pred = (prob >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    accuracy = (tp + tn) / max(len(y_test), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    feature_importance = sorted(
        [{"feature": name, "coef": float(coef)} for name, coef in zip(feature_cols, beta[1:])],
        key=lambda item: abs(item["coef"]),
        reverse=True,
    )

    result = {
        "model": "numpy_logistic_regression",
        "sample_rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "feature_importance": feature_importance,
    }
    (PROCESSED_DIR / "model_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(feature_importance).to_csv(PROCESSED_DIR / "model_feature_importance.csv", index=False, encoding="utf-8-sig")
    return result


def format_int(value: float) -> str:
    return f"{int(round(value)):,}"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}\\%"


def main() -> None:
    ensure_dirs()
    results = aggregate_raw_data()
    model_result = train_logistic_model(results["user_features"])
    build_charts(results, model_result)

    summary = {
        "meta": results["meta"],
        "event_summary": results["event_summary"].to_dict(orient="records"),
        "funnel": results["funnel"].to_dict(orient="records"),
        "model": model_result,
    }
    (PROCESSED_DIR / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFinished.")
    print(json.dumps(summary["meta"], ensure_ascii=False, indent=2))
    print(json.dumps(model_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
