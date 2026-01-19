import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor


TARGET_COLS = ["real_weight", "real_height", "real_width", "real_length"]
OUTPUT_COLS = ["weight", "height", "length", "width"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["description"]).str.strip()

    dates = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_year"] = dates.dt.year.fillna(0).astype(int)
    df["order_month"] = dates.dt.month.fillna(0).astype(int)
    df["order_dow"] = dates.dt.dayofweek.fillna(0).astype(int)

    df["item_price"] = df["item_price"].fillna(0.0).clip(lower=0)
    df["item_price_log"] = np.log1p(df["item_price"])

    df["title_len"] = df["title"].str.len()
    df["desc_len"] = df["description"].str.len()

    return df


def load_image(path: str) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    return img


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_feats = model.get_text_features(**inputs)
            text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
            features.append(text_feats.cpu().numpy())
    return np.vstack(features)


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [load_image(p) for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feats = model.get_image_features(**inputs)
            img_feats = torch.nn.functional.normalize(img_feats, dim=-1)
            features.append(img_feats.cpu().numpy())
    return np.vstack(features)


def build_features(
    df: pd.DataFrame,
    image_dir: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    texts = df["text"].tolist()
    image_paths = [
        os.path.join(image_dir, name) if isinstance(name, str) else ""
        for name in df["image_name"].tolist()
    ]

    text_emb = encode_texts(model, processor, texts, device, batch_size)
    image_emb = encode_images(model, processor, image_paths, device, batch_size)

    numeric_cols = ["item_price_log", "order_year", "order_month", "order_dow", "title_len", "desc_len"]
    numeric = df[numeric_cols].to_numpy(dtype=np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0)

    features = np.concatenate([text_emb, image_emb, numeric], axis=1)
    return features


def log_mae_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.nan_to_num(y_true, nan=0.0)
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    per_target = np.mean(np.abs(log_true - log_pred), axis=0)
    return dict(zip(TARGET_COLS, per_target))


def macro_log_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_target = log_mae_per_target(y_true, y_pred)
    return float(np.mean(list(per_target.values())))


def weight_bucket(weight: float) -> str:
    if weight != weight:
        return "unknown"
    if weight < 2:
        return "<2kg"
    if weight < 5:
        return "2-5kg"
    if weight < 10:
        return "5-10kg"
    if weight < 20:
        return "10-20kg"
    return "20kg+"


def eval_group_metrics(valid_df: pd.DataFrame, y_pred: np.ndarray) -> None:
    y_true = valid_df[TARGET_COLS].to_numpy()
    valid_df = valid_df.copy()
    valid_df["weight_bucket"] = valid_df["real_weight"].map(weight_bucket)

    group_cols = [
        "category_name",
        "subcategory_name",
        "microcat_name",
        "weight_bucket",
    ]

    for col in group_cols:
        grouped = valid_df.groupby(col).indices
        metrics = []
        for key, idx in grouped.items():
            if len(idx) < 50:
                continue
            score = macro_log_mae(y_true[idx], y_pred[idx])
            metrics.append((key, len(idx), score))
        metrics.sort(key=lambda x: x[2], reverse=True)
        print(f"Top {col} Macro Log-MAE (worst 10):")
        for key, size, score in metrics[:10]:
            print(f"  {key}: n={size}, macro_log_mae={score:.4f}")


def train_and_predict(
    train_path: str,
    test_path: str,
    train_image_dir: str,
    test_image_dir: str,
    output_path: str,
    valid_size: float,
    model_name: str,
    batch_size: int,
):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)

    train_features = build_features(
        train_df, train_image_dir, clip_model, clip_processor, device, batch_size
    )
    test_features = build_features(
        test_df, test_image_dir, clip_model, clip_processor, device, batch_size
    )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    model = MultiOutputRegressor(Ridge(alpha=1.0))

    if valid_size > 0:
        X_train, X_valid, y_train, y_valid, df_train, df_valid = train_test_split(
            train_features,
            train_df[TARGET_COLS].to_numpy(),
            train_df,
            test_size=valid_size,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        per_target = log_mae_per_target(y_valid, preds)
        print("Validation Macro Log-MAE:", macro_log_mae(y_valid, preds))
        print("Validation Log-MAE per target:", per_target)
        eval_group_metrics(df_valid, preds)
    else:
        model.fit(train_features, train_df[TARGET_COLS].to_numpy())

    test_preds = model.predict(test_features)
    submission = pd.DataFrame(test_preds, columns=["weight", "height", "width", "length"])
    submission = submission[OUTPUT_COLS]
    submission.insert(0, "item_id", test_df["item_id"].values)
    submission.to_csv(output_path, index=False)


if __name__ == "__main__":
    train_path = "train.parquet"
    test_path = "test.parquet"
    train_images = "train"
    test_images = "test"
    output_path = "submission.csv"
    valid_size = 0.0
    model_name = "openai/clip-vit-base-patch32"
    batch_size = 64

    train_and_predict(
        train_path,
        test_path,
        train_images,
        test_images,
        output_path,
        valid_size,
        model_name,
        batch_size,
    )
