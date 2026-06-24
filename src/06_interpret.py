import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance

MODELS_DIR = Path("data/processed/models")
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("output/tables")


def main() -> None:
    best_model_name = (TABLES_DIR / "best_model.txt").read_text(encoding="utf-8").strip()
    model = joblib.load(MODELS_DIR / f"{best_model_name}.joblib")

    train_df = pd.read_csv(PROCESSED_DIR / "model_input_train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "model_input_test.csv")
    predictions = pd.read_csv(PROCESSED_DIR / "predictions.csv")

    feature_names = json.loads((TABLES_DIR / "feature_names.json").read_text(encoding="utf-8"))
    X_test = test_df[feature_names]
    y_test = test_df["log_price_rub"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    mean_abs_shap.to_frame("mean_abs_shap").to_csv(TABLES_DIR / "shap_importance.csv")
    print("Top-10 SHAP importance:\n", mean_abs_shap.head(10))

    np.save(PROCESSED_DIR / "shap_values_test.npy", shap_values)

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importance = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    perm_importance.to_frame("perm_importance_mean").to_csv(TABLES_DIR / "permutation_importance.csv")

    top_shap_5 = set(mean_abs_shap.head(5).index)
    top_perm_5 = set(perm_importance.head(5).index)
    overlap = len(top_shap_5 & top_perm_5)
    print(f"\nSHAP vs permutation importance: {overlap}/5 совпадений в топ-5.")

    pd_results = {}
    for feat in ["area_sqm", "metro_time_min"]:
        pdp = partial_dependence(model, X_test, [feat], kind="average", grid_resolution=30)
        pd_results[feat] = {"grid": pdp["grid_values"][0].tolist(), "average": pdp["average"][0].tolist()}
    Path(PROCESSED_DIR / "partial_dependence.json").write_text(json.dumps(pd_results), encoding="utf-8")

    # predictions и test_df построены из одного df_test без промежуточного шаффла (см. 04) —
    # поэтому строки совпадают по позиции, можно сопоставлять price_segment напрямую.
    segments = predictions["price_segment"].reset_index(drop=True)
    assert len(segments) == len(X_test)

    h4_lines = ["# H4 — важность `metro_time_min` по ценовым сегментам\n"]
    feat_idx = feature_names.index("metro_time_min")
    for seg in ["econom", "premium"]:
        mask = (segments == seg).values
        seg_shap = np.abs(shap_values[mask, feat_idx]).mean()
        seg_rank = (
            pd.Series(np.abs(shap_values[mask]).mean(axis=0), index=feature_names)
            .sort_values(ascending=False)
            .index.get_loc("metro_time_min")
            + 1
        )
        h4_lines.append(f"- Сегмент `{seg}` (n={mask.sum()}): mean|SHAP(metro_time_min)| = {seg_shap:.4f}, ранг признака = {seg_rank}/{len(feature_names)}\n")

    econom_shap = np.abs(shap_values[(segments == "econom").values, feat_idx]).mean()
    premium_shap = np.abs(shap_values[(segments == "premium").values, feat_idx]).mean()
    confirmed = econom_shap > premium_shap
    h4_lines.append(
        f"\n**H4 {'подтверждается' if confirmed else 'не подтверждается'}**: важность `metro_time_min` "
        f"в сегменте `econom` {'выше' if confirmed else 'не выше'}, чем в `premium` "
        f"({econom_shap:.4f} {'>' if confirmed else '<='} {premium_shap:.4f}).\n"
    )
    (TABLES_DIR / "h4_segment_comparison.md").write_text("\n".join(h4_lines), encoding="utf-8")
    print("\n".join(h4_lines))


if __name__ == "__main__":
    main()
