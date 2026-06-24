import json
from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/cleaned.csv")
OUT_PATH = Path("data/processed/features.csv")
RARE_THRESHOLD = 20

# deposit_rub исключён: corr(deposit_rub, price_rub) = 0.91 (депозит почти всегда = 1 месячной
# плате) — это утечка целевой переменной, а не структурный признак объекта.
NUMERIC_FEATURES = [
    "rooms",
    "area_sqm",
    "living_area_sqm",
    "kitchen_area_sqm",
    "floor",
    "floors_total",
    "metro_time_min",
    "build_year",
    "prepay_months",
    "rooms_was_missing",
    "living_area_sqm_missing",
    "kitchen_area_sqm_missing",
    "build_year_missing",
    "prepay_months_missing",
    "is_agent",
]
CATEGORICAL_FEATURES = ["repair_type", "house_material", "district_grouped", "metro_grouped"]
TARGET_COL = "log_price_rub"
NON_FEATURE_COLS = ["offer_id", "added", "price_rub", "price_per_sqm", "price_segment", "district", "metro"]


def group_rare_categories(series: pd.Series, threshold: int) -> pd.Series:
    counts = series.value_counts()
    rare = counts[counts < threshold].index
    return series.where(~series.isin(rare), other="Other")


def main() -> None:
    df = pd.read_csv(IN_PATH)

    df["price_per_sqm"] = df["price_rub"] / df["area_sqm"]
    # сегмент только для пост-хок анализа (H4: сравнение SHAP по подвыборкам),
    # как признак модели не используется — иначе утечка через цену.
    df["price_segment"] = pd.qcut(
        df["price_per_sqm"], q=4, labels=["econom", "comfort", "business", "premium"]
    )

    n_district_before = df["district"].nunique()
    n_metro_before = df["metro"].nunique()
    df["district_grouped"] = group_rare_categories(df["district"], RARE_THRESHOLD)
    df["metro_grouped"] = group_rare_categories(df["metro"], RARE_THRESHOLD)
    print(
        f"district: {n_district_before} -> {df['district_grouped'].nunique()} категорий "
        f"(порог {RARE_THRESHOLD} объявлений)"
    )
    print(
        f"metro: {n_metro_before} -> {df['metro_grouped'].nunique()} категорий "
        f"(порог {RARE_THRESHOLD} объявлений)"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    contract = {
        "target_col": TARGET_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "non_feature_cols": NON_FEATURE_COLS,
        "price_segment_distribution": df["price_segment"].value_counts().to_dict(),
    }
    out_json = Path("output/tables/feature_columns.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(contract, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"Saved {len(df)} rows -> {OUT_PATH}")
    print(f"Feature contract -> {out_json}")


if __name__ == "__main__":
    main()
