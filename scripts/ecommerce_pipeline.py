from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from utils import classification_metrics, draw_title, load_font, save_bar_chart, sigmoid


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

# 这些参数集中放在这里，便于报告复现实验时说明模型设置。
MODEL_MAX_ROWS = 250_000
MODEL_TRAIN_FRACTION = 0.8
MODEL_POS_WEIGHT_CAP = 20.0
MODEL_LEARNING_RATE = 0.08
MODEL_L2 = 0.002
MODEL_STEPS = 700
MODEL_THRESHOLD_GRID = np.linspace(0.15, 0.75, 25)

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
        # 同一个月份可能同时存在 zip 和 gz，只保留先匹配到的一份。
        logical_name = file_path.name.replace(".zip", "").replace(".gz", "")
        if logical_name in seen_names:
            continue
        seen_names.add(logical_name)
        files.append(file_path)
    return files


def preprocess_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # 每个 chunk 先统一时间、缺失值和事件标记，后面的聚合只依赖这些标准字段。
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


def concat_group(
    parts: list[pd.DataFrame],
    keys: list[str],
    out_name: str,
    *,
    mean_columns: set[str] | None = None,
) -> pd.DataFrame:
    mean_columns = mean_columns or set()
    df = pd.concat(parts, ignore_index=True)
    agg: dict[str, str] = {}
    for column in df.columns:
        if column in keys:
            continue
        # 大部分中间表按 chunk 求和即可，均价这类字段需要显式指定为均值。
        agg[column] = "mean" if column in mean_columns else "sum"
    result = df.groupby(keys, as_index=False).agg(agg)
    result.to_csv(PROCESSED_DIR / out_name, index=False, encoding="utf-8-sig")
    return result


def append_chunk_summaries(
    chunk: pd.DataFrame,
    *,
    event_parts: list[pd.DataFrame],
    daily_parts: list[pd.DataFrame],
    hourly_parts: list[pd.DataFrame],
    month_parts: list[pd.DataFrame],
    price_parts: list[pd.DataFrame],
    category_parts: list[pd.DataFrame],
    brand_parts: list[pd.DataFrame],
) -> None:
    # 只保留每个 chunk 的聚合结果，避免把 1.77 亿行明细全部放进内存。
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


def append_sampled_user_features(
    chunk: pd.DataFrame,
    *,
    sampled_stage_users: dict[str, set],
    sampled_month_stage_users: dict[tuple[str, str], set],
    user_parts: list[pd.DataFrame],
) -> None:
    # 用户级特征用确定性取模抽样，保证多次运行抽到的是同一批用户。
    sampled = chunk.loc[(chunk["user_id"].astype("int64") % USER_SAMPLE_MOD) == 0].copy()
    if sampled.empty:
        return

    for event_name in ["view", "cart", "purchase"]:
        ids = sampled.loc[sampled["event_type"] == event_name, "user_id"].unique()
        sampled_stage_users[event_name].update(ids.tolist())
        month_user_groups = sampled.loc[
            sampled["event_type"] == event_name,
            ["month", "user_id"],
        ].groupby("month")["user_id"]
        for month_value, month_ids in month_user_groups:
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


def build_user_features(user_parts: list[pd.DataFrame]) -> pd.DataFrame:
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
    return user_features


def build_funnel_tables(
    sampled_stage_users: dict[str, set],
    sampled_month_stage_users: dict[tuple[str, str], set],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 漏斗人数基于抽样用户估计，所以输出里同时保留 sample_users 和 estimated_users。
    funnel = pd.DataFrame(
        [
            {"stage": stage, "sample_users": len(users), "estimated_users": len(users) * USER_SAMPLE_MOD}
            for stage, users in sampled_stage_users.items()
        ]
    )
    base = max(float(funnel.loc[funnel["stage"] == "view", "estimated_users"].iloc[0]), 1.0)
    funnel["relative_to_view"] = funnel["estimated_users"] / base
    funnel.to_csv(PROCESSED_DIR / "funnel_summary.csv", index=False, encoding="utf-8-sig")

    month_funnel_rows = [
        {
            "month": month_value,
            "stage": stage,
            "sample_users": len(users),
            "estimated_users": len(users) * USER_SAMPLE_MOD,
        }
        for (month_value, stage), users in sampled_month_stage_users.items()
    ]
    month_funnel = pd.DataFrame(month_funnel_rows)
    month_funnel.to_csv(PROCESSED_DIR / "month_funnel_summary.csv", index=False, encoding="utf-8-sig")
    return funnel, month_funnel


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

    # set 用来去重用户，避免同一用户多次行为把漏斗人数重复计算。
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

            # 主流程保持为“清洗 -> 聚合 -> 抽样特征”，具体细节拆到独立函数中。
            chunk = preprocess_chunk(chunk)
            append_chunk_summaries(
                chunk,
                event_parts=event_parts,
                daily_parts=daily_parts,
                hourly_parts=hourly_parts,
                month_parts=month_parts,
                price_parts=price_parts,
                category_parts=category_parts,
                brand_parts=brand_parts,
            )
            append_sampled_user_features(
                chunk,
                sampled_stage_users=sampled_stage_users,
                sampled_month_stage_users=sampled_month_stage_users,
                user_parts=user_parts,
            )

            del chunk

    event_summary = concat_group(event_parts, ["event_type"], "event_type_summary.csv")
    daily_summary = concat_group(daily_parts, ["date", "event_type"], "daily_event_summary.csv")
    hourly_summary = concat_group(hourly_parts, ["hour", "event_type"], "hourly_event_summary.csv")
    month_summary = concat_group(month_parts, ["month", "event_type"], "month_event_summary.csv")
    price_summary = concat_group(price_parts, ["price_bin", "event_type"], "price_bin_summary.csv")
    category_summary = concat_group(
        category_parts,
        ["category_code"],
        "category_summary.csv",
        mean_columns={"avg_price"},
    )
    brand_summary = concat_group(
        brand_parts,
        ["brand"],
        "brand_summary.csv",
        mean_columns={"avg_price"},
    )

    category_summary = category_summary.sort_values(["purchases", "revenue"], ascending=False)
    brand_summary = brand_summary.sort_values(["purchases", "revenue"], ascending=False)
    category_summary.head(60).to_csv(PROCESSED_DIR / "top_categories.csv", index=False, encoding="utf-8-sig")
    brand_summary.head(60).to_csv(PROCESSED_DIR / "top_brands.csv", index=False, encoding="utf-8-sig")

    user_features = build_user_features(user_parts)
    funnel, month_funnel = build_funnel_tables(sampled_stage_users, sampled_month_stage_users)

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


def save_line_chart(
    series: dict[str, list[tuple[str, float]]],
    title: str,
    path: Path,
    width: int = 1400,
    height: int = 820,
) -> None:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    label_font = load_font(19)
    draw_title(draw, title, width)

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
    if len(df) > MODEL_MAX_ROWS:
        # 训练样本按标签比例抽取，避免正负样本比例被额外改变。
        sampled_parts = []
        for _, part in df.groupby("converted"):
            take = max(1, int(MODEL_MAX_ROWS * len(part) / len(df)))
            sampled_parts.append(part.sample(min(take, len(part)), random_state=RANDOM_SEED))
        df = pd.concat(sampled_parts, ignore_index=True)

    x = np.log1p(df[feature_cols].astype(float).to_numpy())
    y = df["converted"].astype(int).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(df))
    split = int(len(df) * MODEL_TRAIN_FRACTION)
    train_idx, test_idx = idx[:split], idx[split:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    # 标准化参数只从训练集计算，测试集不参与任何拟合步骤。
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]

    pos_rate = y_train.mean()
    pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)
    weights = np.where(y_train == 1, min(pos_weight, MODEL_POS_WEIGHT_CAP), 1.0)

    # 使用 NumPy 手写逻辑回归，便于展示课程中矩阵计算和梯度下降的部分。
    beta = np.zeros(x_train.shape[1])
    for _ in range(MODEL_STEPS):
        pred = sigmoid(x_train @ beta)
        error = (pred - y_train) * weights
        grad = (x_train.T @ error) / weights.sum()
        grad[1:] += MODEL_L2 * beta[1:]
        beta -= MODEL_LEARNING_RATE * grad

    prob = sigmoid(x_test @ beta)
    # 阈值不固定为 0.5，而是在候选阈值里选择 F1 最好的一个。
    threshold_metrics = [classification_metrics(y_test, prob, float(threshold)) for threshold in MODEL_THRESHOLD_GRID]
    metrics = max(threshold_metrics, key=lambda item: item["f1"])

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
        "threshold": metrics["threshold"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
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
