import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = Path("output/figures")
TABLES_DIR = Path("output/tables")
PROCESSED_DIR = Path("data/processed")

sns.set_theme(style="whitegrid")


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    predictions = pd.read_csv(PROCESSED_DIR / "predictions.csv")
    best_model = (TABLES_DIR / "best_model.txt").read_text(encoding="utf-8").strip()
    residuals = pd.read_csv(PROCESSED_DIR / "best_model_residuals.csv")
    shap_importance = pd.read_csv(TABLES_DIR / "shap_importance.csv", index_col=0)
    pd_data = json.loads((PROCESSED_DIR / "partial_dependence.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["price_rub"], bins=60, ax=axes[0])
    axes[0].set_title("Распределение арендной платы, ₽/мес.")
    sns.histplot(np.log1p(df["price_rub"]), bins=60, ax=axes[1])
    axes[1].set_title("log1p(price_rub)")
    savefig("01_price_distribution.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(df["price_per_sqm"].clip(upper=df["price_per_sqm"].quantile(0.99)), bins=60)
    plt.title("Распределение цены за м² (обрезано по 99-му перцентилю)")
    savefig("02_price_per_sqm_distribution.png")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df.sample(min(5000, len(df)), random_state=42), x="area_sqm", y="price_rub", alpha=0.3, s=15)
    plt.title("Арендная плата vs площадь")
    savefig("03_price_vs_area_scatter.png")

    plt.figure(figsize=(7, 4))
    order = ["no", "cosmetic", "euro", "design", "unknown"]
    sns.boxplot(data=df, x="repair_type", y="price_per_sqm", order=order, showfliers=False)
    plt.title("Цена за м² по типу ремонта (H2)")
    savefig("04_price_per_sqm_by_repair.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="house_material", y="price_per_sqm", showfliers=False)
    plt.title("Цена за м² по материалу дома")
    plt.xticks(rotation=30, ha="right")
    savefig("05_price_per_sqm_by_material.png")

    top_districts = (
        df.groupby("district")["price_per_sqm"].mean().sort_values(ascending=False).head(15)
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_districts.values, y=top_districts.index, color="steelblue")
    plt.title("Топ-15 районов по средней цене за м²")
    plt.xlabel("₽/м²")
    savefig("06_top_districts_price_per_sqm.png")

    top_shap = shap_importance["mean_abs_shap"].sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_shap.values, y=top_shap.index, color="darkorange")
    plt.title(f"Важность признаков (mean |SHAP|), модель {best_model}")
    savefig("07_shap_feature_importance.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(predictions["price_rub"], predictions[f"{best_model}_pred_rub"], alpha=0.3, s=10)
    lims = [0, predictions["price_rub"].quantile(0.99)]
    plt.plot(lims, lims, color="red", linestyle="--")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Фактическая цена, ₽/мес.")
    plt.ylabel("Прогноз, ₽/мес.")
    plt.title(f"Predicted vs Actual ({best_model})")
    savefig("08_predicted_vs_actual.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(residuals["pred_rub"], residuals["residual_rub"], alpha=0.3, s=10)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Прогноз, ₽/мес.")
    plt.ylabel("Остаток, ₽/мес.")
    plt.title("Остатки модели vs прогноз")
    savefig("09_residuals_vs_predicted.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(pd_data["area_sqm"]["grid"], pd_data["area_sqm"]["average"])
    axes[0].set_title("Partial dependence: area_sqm")
    axes[0].set_xlabel("м²")
    axes[0].set_ylabel("log_price_rub (предсказание)")
    axes[1].plot(pd_data["metro_time_min"]["grid"], pd_data["metro_time_min"]["average"])
    axes[1].set_title("Partial dependence: metro_time_min")
    axes[1].set_xlabel("мин. до метро")
    savefig("10_partial_dependence.png")

    print(f"Saved {len(list(FIG_DIR.glob('*.png')))} figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
