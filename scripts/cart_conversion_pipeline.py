from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ecommerce"
PROCESSED_DIR = ROOT / "data" / "processed" / "ecommerce"
FIG_DIR = ROOT / "outputs" / "figures"

FILES = [
    RAW_DIR / "2019-Oct.csv.gz",
    RAW_DIR / "2019-Nov.csv.zip",
    RAW_DIR / "2019-Dec.csv.gz",
]

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

CHUNKSIZE = 1_000_000
USER_SAMPLE_MOD = 20
RANDOM_SEED = 2026


@dataclass
class SessionState:
    user_id: int
    user_session: str
    first_event_time: pd.Timestamp | None = None
    first_cart_time: pd.Timestamp | None = None
    first_cart_price: float = 0.0
    first_cart_product: str = ""
    first_cart_category: str = ""
    first_cart_brand: str = ""
    cart_hour: int = 0
    cart_weekday: int = 0
    pre_events: int = 0
    pre_views: int = 0
    pre_purchases: int = 0
    pre_price_sum: float = 0.0
    pre_price_count: int = 0
    pre_max_price: float = 0.0
    pre_products: set[str] = field(default_factory=set)
    pre_categories: set[str] = field(default_factory=set)
    pre_brands: set[str] = field(default_factory=set)
    pre_product_view_counts: dict[str, int] = field(default_factory=dict)
    pre_category_view_counts: dict[str, int] = field(default_factory=dict)
    same_product_viewed_before_cart: int = 0
    same_category_viewed_before_cart: int = 0
    first_cart_product_view_count_before_cart: int = 0
    first_cart_category_view_count_before_cart: int = 0
    any_purchase_after_cart: int = 0
    same_product_purchase_after_cart: int = 0

    def update_before_cart(
        self,
        event_time: pd.Timestamp,
        event_type: str,
        product_id: str,
        category_code: str,
        brand: str,
        price: float,
    ) -> None:
        if self.first_event_time is None:
            self.first_event_time = event_time
        self.pre_events += 1
        if event_type == "view":
            self.pre_views += 1
            self.pre_product_view_counts[product_id] = self.pre_product_view_counts.get(product_id, 0) + 1
            self.pre_category_view_counts[category_code] = self.pre_category_view_counts.get(category_code, 0) + 1
        if event_type == "purchase":
            self.pre_purchases += 1
        self.pre_price_sum += price
        self.pre_price_count += 1
        self.pre_max_price = max(self.pre_max_price, price)
        self.pre_products.add(product_id)
        self.pre_categories.add(category_code)
        self.pre_brands.add(brand)

    def set_first_cart(
        self,
        event_time: pd.Timestamp,
        product_id: str,
        category_code: str,
        brand: str,
        price: float,
    ) -> None:
        if self.first_event_time is None:
            self.first_event_time = event_time
        self.first_cart_time = event_time
        self.first_cart_price = price
        self.first_cart_product = product_id
        self.first_cart_category = category_code
        self.first_cart_brand = brand
        self.cart_hour = int(event_time.hour)
        self.cart_weekday = int(event_time.weekday())
        self.same_product_viewed_before_cart = int(product_id in self.pre_products)
        self.same_category_viewed_before_cart = int(category_code in self.pre_categories)
        self.first_cart_product_view_count_before_cart = self.pre_product_view_counts.get(product_id, 0)
        self.first_cart_category_view_count_before_cart = self.pre_category_view_counts.get(category_code, 0)

    def to_record(self) -> dict:
        assert self.first_cart_time is not None
        duration = (self.first_cart_time - self.first_event_time).total_seconds() / 60.0 if self.first_event_time else 0.0
        avg_price = self.pre_price_sum / max(self.pre_price_count, 1)
        return {
            "month": self.first_cart_time.strftime("%Y-%m"),
            "user_id": self.user_id,
            "user_session": self.user_session,
            "first_cart_time": self.first_cart_time.strftime("%Y-%m-%d %H:%M:%S"),
            "label_any_purchase_after_cart": self.any_purchase_after_cart,
            "label_same_product_purchase_after_cart": self.same_product_purchase_after_cart,
            "pre_cart_events": self.pre_events,
            "pre_cart_views": self.pre_views,
            "pre_cart_purchases": self.pre_purchases,
            "pre_cart_unique_products": len(self.pre_products),
            "pre_cart_unique_categories": len(self.pre_categories),
            "pre_cart_unique_brands": len(self.pre_brands),
            "first_cart_price": self.first_cart_price,
            "avg_pre_cart_price": avg_price,
            "max_pre_cart_price": self.pre_max_price,
            "session_duration_before_cart_minutes": max(duration, 0.0),
            "cart_hour": self.cart_hour,
            "cart_weekday": self.cart_weekday,
            "same_product_viewed_before_cart": self.same_product_viewed_before_cart,
            "same_category_viewed_before_cart": self.same_category_viewed_before_cart,
            "first_cart_product_view_count_before_cart": self.first_cart_product_view_count_before_cart,
            "first_cart_category_view_count_before_cart": self.first_cart_category_view_count_before_cart,
        }


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_cart_session_table() -> pd.DataFrame:
    out_path = PROCESSED_DIR / "cart_conversion_sessions.csv"
    if out_path.exists():
        print(f"Using cached session table: {out_path}")
        return pd.read_csv(out_path)

    states: dict[tuple[int, str], SessionState] = {}
    total_rows = 0
    sampled_rows = 0
    t0 = time.time()

    for file_path in FILES:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        print(f"\nProcessing {file_path.name}: {file_path.stat().st_size / 1024 ** 3:.2f} GB")
        reader = pd.read_csv(
            file_path,
            usecols=USECOLS,
            chunksize=CHUNKSIZE,
            compression="infer",
            low_memory=False,
        )
        for chunk_id, chunk in enumerate(reader, start=1):
            total_rows += len(chunk)
            chunk = chunk.dropna(subset=["user_id", "user_session", "event_time"])
            chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")
            chunk = chunk.dropna(subset=["user_id"])
            chunk["user_id"] = chunk["user_id"].astype("int64")
            chunk = chunk.loc[(chunk["user_id"] % USER_SAMPLE_MOD) == 0].copy()
            if len(chunk) == 0:
                continue

            sampled_rows += len(chunk)
            chunk["event_time"] = pd.to_datetime(chunk["event_time"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["event_time"]).sort_values("event_time")
            chunk["product_id"] = chunk["product_id"].fillna("unknown").astype(str)
            chunk["category_code"] = chunk["category_code"].fillna("unknown").astype(str)
            chunk["brand"] = chunk["brand"].fillna("unknown").astype(str)
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce").fillna(0).clip(lower=0)
            chunk["user_session"] = chunk["user_session"].astype(str)

            for row in chunk.itertuples(index=False):
                key = (int(row.user_id), row.user_session)
                state = states.get(key)
                if state is None:
                    state = SessionState(user_id=int(row.user_id), user_session=row.user_session)
                    states[key] = state

                event_type = str(row.event_type)
                if state.first_cart_time is None:
                    if event_type == "cart":
                        state.set_first_cart(
                            row.event_time,
                            row.product_id,
                            row.category_code,
                            row.brand,
                            float(row.price),
                        )
                    else:
                        state.update_before_cart(
                            row.event_time,
                            event_type,
                            row.product_id,
                            row.category_code,
                            row.brand,
                            float(row.price),
                        )
                elif event_type == "purchase":
                    state.any_purchase_after_cart = 1
                    if row.product_id == state.first_cart_product:
                        state.same_product_purchase_after_cart = 1

            elapsed = (time.time() - t0) / 60
            print(
                f"  chunk {chunk_id:03d}, total_rows={total_rows:,}, "
                f"sampled_rows={sampled_rows:,}, states={len(states):,}, elapsed={elapsed:.1f} min",
                flush=True,
            )

    records = [state.to_record() for state in states.values() if state.first_cart_time is not None]
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    meta = {
        "total_rows_scanned": int(total_rows),
        "sampled_rows": int(sampled_rows),
        "user_sample_mod": USER_SAMPLE_MOD,
        "cart_sessions": int(len(df)),
        "any_purchase_positive_rate": float(df["label_any_purchase_after_cart"].mean()) if len(df) else 0.0,
        "same_product_positive_rate": float(df["label_same_product_purchase_after_cart"].mean()) if len(df) else 0.0,
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
    }
    (PROCESSED_DIR / "cart_conversion_sample_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return df


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_aug = np.c_[np.ones(len(x_train)), x_train]
    beta = np.zeros(x_aug.shape[1])
    pos_rate = max(float(y_train.mean()), 1e-6)
    pos_weight = min((1 - pos_rate) / pos_rate, 20.0)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)
    lr = 0.08
    reg = 0.002
    for _ in range(800):
        pred = sigmoid(x_aug @ beta)
        err = (pred - y_train) * sample_weight
        grad = (x_aug.T @ err) / sample_weight.sum()
        grad[1:] += reg * beta[1:]
        beta -= lr * grad
    return beta


def predict_logistic(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(np.c_[np.ones(len(x)), x] @ beta)


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
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


def topk_table(y_true: np.ndarray, scores: np.ndarray, model_name: str) -> list[dict]:
    base_rate = max(float(y_true.mean()), 1e-9)
    order = np.argsort(-scores)
    rows = []
    total_pos = max(int(y_true.sum()), 1)
    for frac in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(len(y_true) * frac))
        selected = y_true[order[:k]]
        precision = float(selected.mean())
        recall = float(selected.sum() / total_pos)
        rows.append(
            {
                "model": model_name,
                "top_fraction": frac,
                "top_n": int(k),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "lift": precision / base_rate,
            }
        )
    return rows


def low_probability_high_value(test_df: pd.DataFrame, prob: np.ndarray) -> dict:
    tmp = test_df.copy()
    tmp["predicted_purchase_probability"] = prob
    price_cut = float(tmp["first_cart_price"].quantile(0.50))
    prob_cut = float(tmp["predicted_purchase_probability"].quantile(0.20))
    selected = tmp.loc[
        (tmp["first_cart_price"] >= price_cut)
        & (tmp["predicted_purchase_probability"] <= prob_cut)
    ]
    return {
        "definition": "first_cart_price >= test median and predicted natural conversion probability <= bottom 20%",
        "price_cut": price_cut,
        "probability_cut": prob_cut,
        "sessions": int(len(selected)),
        "actual_any_purchase_rate": float(selected["label_any_purchase_after_cart"].mean()) if len(selected) else 0.0,
        "avg_first_cart_price": float(selected["first_cart_price"].mean()) if len(selected) else 0.0,
    }


def train_and_evaluate(df: pd.DataFrame) -> dict:
    feature_cols = [
        "pre_cart_events",
        "pre_cart_views",
        "pre_cart_unique_products",
        "pre_cart_unique_categories",
        "pre_cart_unique_brands",
        "first_cart_product_view_count_before_cart",
        "first_cart_category_view_count_before_cart",
        "first_cart_price",
        "avg_pre_cart_price",
        "max_pre_cart_price",
        "session_duration_before_cart_minutes",
        "cart_hour",
        "cart_weekday",
        "same_product_viewed_before_cart",
        "same_category_viewed_before_cart",
    ]
    label_col = "label_any_purchase_after_cart"
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + [label_col])
    df["month"] = df["month"].astype(str)

    train_df = df.loc[df["month"].isin(["2019-10", "2019-11"])].copy()
    test_df = df.loc[df["month"] == "2019-12"].copy()
    if min(len(train_df), len(test_df)) == 0:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.permutation(len(df))
        train_end = int(len(df) * 0.8)
        train_df = df.iloc[idx[:train_end]].copy()
        test_df = df.iloc[idx[train_end:]].copy()

    x_train_raw = np.log1p(train_df[feature_cols].astype(float).to_numpy())
    x_test_raw = np.log1p(test_df[feature_cols].astype(float).to_numpy())
    y_train = train_df[label_col].astype(int).to_numpy()
    y_test = test_df[label_col].astype(int).to_numpy()

    mean = x_train_raw.mean(axis=0)
    std = x_train_raw.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train_raw - mean) / std
    x_test = (x_test_raw - mean) / std

    beta = fit_logistic(x_train, y_train)
    train_prob = predict_logistic(beta, x_train)
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = classification_metrics(y_train, train_prob, float(threshold))
        candidates.append((metrics["f1"], threshold))
    threshold = float(max(candidates)[1])
    test_prob = predict_logistic(beta, x_test)
    metrics = classification_metrics(y_test, test_prob, threshold)

    rule_score = (
        np.log1p(test_df["pre_cart_views"].to_numpy(dtype=float))
        + 0.6 * np.log1p(test_df["first_cart_product_view_count_before_cart"].to_numpy(dtype=float))
        + 0.8 * test_df["same_product_viewed_before_cart"].to_numpy(dtype=float)
        + 0.4 * np.log1p(test_df["session_duration_before_cart_minutes"].to_numpy(dtype=float))
    )
    topk_rows = topk_table(y_test, rule_score, "rule_pre_cart_interest") + topk_table(
        y_test, test_prob, "logistic_regression"
    )
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(PROCESSED_DIR / "cart_conversion_topk.csv", index=False, encoding="utf-8-sig")

    feature_importance = sorted(
        [{"feature": name, "coef": float(coef)} for name, coef in zip(feature_cols, beta[1:])],
        key=lambda item: abs(item["coef"]),
        reverse=True,
    )
    pd.DataFrame(feature_importance).to_csv(
        PROCESSED_DIR / "cart_conversion_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "task": "cart_to_any_purchase_after_first_cart",
        "main_label": label_col,
        "auxiliary_label": "label_same_product_purchase_after_cart",
        "sample_rows": int(len(df)),
        "train_month": "2019-10 and 2019-11",
        "test_month": "2019-12",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "threshold_selection": "best F1 on training months; TopK/Lift is the primary business evaluation",
        "features": feature_cols,
        "model": "numpy_logistic_regression",
        "metrics": metrics,
        "feature_importance": feature_importance,
        "topk": topk_rows,
        "same_product_positive_rate_test": float(test_df["label_same_product_purchase_after_cart"].mean()),
        "low_probability_high_value": low_probability_high_value(test_df, test_prob),
        "baseline_note": "rule_pre_cart_interest = log1p(pre_cart_views) + first-cart-product view count + same-product signal + session-duration signal",
    }
    (PROCESSED_DIR / "cart_conversion_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def load_font(size: int, bold: bool = False):
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


def save_bar_chart(data: list[tuple[str, float]], title: str, path: Path, color=(42, 111, 151), unit="") -> None:
    width, height = 1280, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((52, 28), title, fill=(25, 32, 45), font=load_font(34, True))
    draw.line((52, 82, width - 52, 82), fill=(211, 218, 229), width=2)
    left, top, right, bottom = 300, 125, width - 80, height - 95
    max_v = max([v for _, v in data] or [1])
    bar_h = max(28, int((bottom - top - 14 * (len(data) - 1)) / max(len(data), 1)))
    for i, (label, value) in enumerate(data):
        y = top + i * (bar_h + 14)
        bar_w = int((right - left) * value / max_v) if max_v else 0
        draw.rounded_rectangle((left, y, left + bar_w, y + bar_h), radius=5, fill=color)
        draw.text((52, y + 3), label[:24], fill=(45, 55, 72), font=load_font(20))
        draw.text((left + bar_w + 10, y + 3), f"{value:.2f}{unit}", fill=(45, 55, 72), font=load_font(19))
    img.save(path)


def save_lift_chart(topk_rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(topk_rows)
    model_df = df.loc[df["model"] == "logistic_regression"].sort_values("top_fraction")
    data = [(f"Top {int(row.top_fraction * 100)}%", float(row.lift)) for row in model_df.itertuples()]
    save_bar_chart(data, "购物车转化预测 Lift@K", path, color=(213, 128, 55), unit="x")


def build_figures(result: dict) -> None:
    feature_data = [(item["feature"], abs(float(item["coef"]))) for item in result["feature_importance"][:10]]
    save_bar_chart(
        feature_data,
        "购物车转化模型特征重要性",
        FIG_DIR / "cart_conversion_feature_importance.png",
        color=(37, 132, 132),
    )
    save_lift_chart(result["topk"], FIG_DIR / "cart_conversion_lift.png")

    for stem in ["cart_conversion_feature_importance", "cart_conversion_lift"]:
        png_path = FIG_DIR / f"{stem}.png"
        jpg_path = FIG_DIR / f"{stem}.jpg"
        Image.open(png_path).convert("RGB").save(jpg_path, quality=95, optimize=True)


def main() -> None:
    ensure_dirs()
    df = build_cart_session_table()
    result = train_and_evaluate(df)
    build_figures(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
