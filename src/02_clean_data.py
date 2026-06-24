import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/moscow_rent_all.csv")
OUT_PATH = Path("data/processed/cleaned.csv")
REPORT_PATH = Path("output/tables/cleaning_report.md")

# >97% NaN в исходных данных (см. data_audit.md) — бесполезны как признаки.
NEAR_EMPTY_COLS = [
    "heating_type",
    "ceiling_height_m",
    "parking_type",
    "passenger_lifts_count",
]

# тексты/идентификаторы/изображения/константы — не нужны табличной модели.
UNUSED_COLS = [
    "url",
    "title",
    "description",
    "detail_features",
    "image_urls",
    "image_paths",
    "images_count",
    "deal_type",
    "price_formatted",
    "repair",
    "address",
    "residential_complex",
    "loggias_count",
    "balconies_count",
    "combined_wcs_count",
    "separate_wcs_count",
    "pets_allowed",
    "children_allowed",
    "utilities_included",
    "details_fetched",
]


def main() -> None:
    df = pd.read_csv(RAW_PATH, sep=";", low_memory=False)
    n_raw = len(df)
    report = [f"# Отчёт об очистке данных\n", f"Исходных строк: {n_raw}\n"]

    df = df.drop(columns=NEAR_EMPTY_COLS + UNUSED_COLS, errors="ignore")
    report.append(
        f"Удалены почти пустые столбцы: {NEAR_EMPTY_COLS}\n"
        f"Удалены неиспользуемые столбцы (тексты/идентификаторы/константы): {UNUSED_COLS}\n"
    )

    # пропуски в rooms: у 81% в описании слово "студия", медианная площадь 24 м² -> студия (0 комнат).
    n_missing_rooms = df["rooms"].isnull().sum()
    df["rooms_was_missing"] = df["rooms"].isnull().astype(int)
    df["rooms"] = df["rooms"].fillna(0)
    report.append(f"`rooms`: {n_missing_rooms} пропусков импутированы как 0 (студия), добавлен флаг `rooms_was_missing`.\n")

    # всего 0.3% строк без метро — отбрасываем, а не имитируем значение.
    n_before = len(df)
    df = df.dropna(subset=["metro", "metro_time_min"])
    report.append(f"Удалены строки без станции метро: {n_before - len(df)} (0.3% датасета).\n")

    for col in ["living_area_sqm", "kitchen_area_sqm", "build_year"]:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
        medians = df.groupby("rooms")[col].transform("median")
        df[col] = df[col].fillna(medians).fillna(df[col].median())
        report.append(f"`{col}`: пропуски заменены медианой по группе `rooms`, добавлен флаг `{col}_missing`.\n")

    for col in ["house_material", "repair_type"]:
        df[col] = df[col].fillna("unknown")
    report.append("`house_material`, `repair_type`: пропуски заменены явной категорией 'unknown'.\n")

    # author_type размечен только для агентов, остальное NaN -> бинарный флаг вместо категории.
    df["is_agent"] = (df["author_type"] == "agent").astype(int)
    df = df.drop(columns=["author_type", "author"], errors="ignore")
    report.append("`author_type`/`author` заменены бинарным флагом `is_agent` (только 'agent' был размечен явно).\n")

    for col in ["deposit_rub", "prepay_months"]:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(df[col].median())
    report.append("`deposit_rub`, `prepay_months`: пропуски (~8-10%) заменены медианой, добавлены флаги missing.\n")

    # отсечение выбросов по 1-99 перцентилю — в данных встречаются явные ошибки ввода цены/площади.
    n_before = len(df)
    for col in ["price_rub", "area_sqm"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    report.append(
        f"Отсечены выбросы по 1-99 перцентилю `price_rub` и `area_sqm`: удалено {n_before - len(df)} строк "
        f"({(n_before - len(df)) / n_before:.1%}).\n"
    )

    # цена сильно скошена -> моделируем log1p(price_rub), а не саму цену.
    df["log_price_rub"] = np.log1p(df["price_rub"])
    report.append(
        f"Добавлена `log_price_rub` = log1p(price_rub) как целевая переменная для моделирования "
        f"(skew price_rub={df['price_rub'].skew():.2f}, skew log_price_rub={df['log_price_rub'].skew():.2f}).\n"
    )

    report.append(f"\nИтоговых строк: {len(df)} (было {n_raw}, потеряно {n_raw - len(df)} = {(n_raw - len(df)) / n_raw:.1%}).\n")
    report.append(f"\nИтоговые столбцы ({len(df.columns)}): {sorted(df.columns.tolist())}\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {OUT_PATH}")
    print(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
