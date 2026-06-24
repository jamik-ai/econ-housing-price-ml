from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PRED_PATH = Path("data/processed/predictions.csv")
METRICS_PATH = Path("output/tables/model_metrics.csv")
DIAG_PATH = Path("output/tables/residual_diagnostics.md")
MODEL_NAMES = ["ridge", "random_forest", "lightgbm"]


def main() -> None:
    df = pd.read_csv(PRED_PATH)
    y_true_rub = df["price_rub"].values
    y_true_log = df["log_price_rub_actual"].values

    rows = []
    for name in MODEL_NAMES:
        pred_rub = df[f"{name}_pred_rub"].values
        pred_log = df[f"{name}_pred_log"].values
        rows.append(
            {
                "model": name,
                "RMSE_rub": np.sqrt(mean_squared_error(y_true_rub, pred_rub)),
                "MAE_rub": mean_absolute_error(y_true_rub, pred_rub),
                # RMSLE на цене = RMSE на log1p(цены), целевая переменная моделей уже в log-шкале.
                "RMSLE": np.sqrt(mean_squared_error(y_true_log, pred_log)),
                "R2_rub": r2_score(y_true_rub, pred_rub),
                "R2_log": r2_score(y_true_log, pred_log),
            }
        )
    metrics = pd.DataFrame(rows).sort_values("RMSLE")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    print(metrics.to_string(index=False))

    best_model = metrics.iloc[0]["model"]
    pred_rub = df[f"{best_model}_pred_rub"].values
    residuals = y_true_rub - pred_rub
    abs_resid_corr = np.corrcoef(np.abs(residuals), y_true_rub)[0, 1]

    diag = [
        f"# Диагностика остатков — лучшая модель: `{best_model}`\n",
        f"RMSLE = {metrics.iloc[0]['RMSLE']:.4f}, R² (руб.) = {metrics.iloc[0]['R2_rub']:.4f}\n",
        f"\nКорреляция |остаток| с фактической ценой: **{abs_resid_corr:.3f}** "
        f"({'есть признаки гетероскедастичности — ошибка растёт с ценой' if abs_resid_corr > 0.2 else 'гетероскедастичность не выражена'}).\n",
        f"\nОстатки (руб.): среднее={residuals.mean():.0f}, std={residuals.std():.0f}, "
        f"median={np.median(residuals):.0f}, P5={np.percentile(residuals, 5):.0f}, P95={np.percentile(residuals, 95):.0f}\n",
    ]
    DIAG_PATH.write_text("\n".join(diag), encoding="utf-8")

    resid_df = df[["offer_id", "price_segment", "price_rub"]].copy()
    resid_df["pred_rub"] = pred_rub
    resid_df["residual_rub"] = residuals
    resid_df.to_csv(Path("data/processed/best_model_residuals.csv"), index=False)
    Path("output/tables/best_model.txt").write_text(best_model, encoding="utf-8")

    print(f"\nBest model: {best_model}")
    print(f"Metrics -> {METRICS_PATH}")
    print(f"Diagnostics -> {DIAG_PATH}")


if __name__ == "__main__":
    main()
