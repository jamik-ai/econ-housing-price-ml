import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("data/processed/models")
PROCESSED_DIR = Path("data/processed")
CONTRACT_PATH = Path("output/tables/feature_columns.json")
RANDOM_STATE = 42
TARGET_ENCODE_SMOOTHING = 20


def target_encode(train_col: pd.Series, train_target: pd.Series, other_col: pd.Series, smoothing: float) -> tuple[pd.Series, pd.Series, dict]:
    # статистика считается только по train и применяется к train/test одинаково —
    # иначе кодирование "видит" test и завышает качество модели.
    global_mean = train_target.mean()
    stats = train_target.groupby(train_col).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    mapping = smoothed.to_dict()
    train_encoded = train_col.map(mapping).fillna(global_mean)
    other_encoded = other_col.map(mapping).fillna(global_mean)
    return train_encoded, other_encoded, {"global_mean": global_mean, "n_categories": len(mapping)}


def build_design_matrix(
    df_train: pd.DataFrame, df_test: pd.DataFrame, numeric_features: list[str], categorical_features: list[str], target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = df_train[numeric_features].copy()
    X_test = df_test[numeric_features].copy()

    # высококардинальные district/metro (133/257 уровней) -> target encoding,
    # а не one-hot (раздуло бы размерность и было бы неустойчиво на редких категориях).
    for col in ["district_grouped", "metro_grouped"]:
        enc_train, enc_test, _ = target_encode(df_train[col], df_train[target_col], df_test[col], TARGET_ENCODE_SMOOTHING)
        X_train[f"{col}_te"] = enc_train.values
        X_test[f"{col}_te"] = enc_test.values

    low_card_cols = [c for c in categorical_features if c not in ("district_grouped", "metro_grouped")]
    train_dummies = pd.get_dummies(df_train[low_card_cols], prefix=low_card_cols)
    test_dummies = pd.get_dummies(df_test[low_card_cols], prefix=low_card_cols)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    X_train = pd.concat([X_train.reset_index(drop=True), train_dummies.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), test_dummies.reset_index(drop=True)], axis=1)
    return X_train, X_test


def main() -> None:
    df = pd.read_csv(FEATURES_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    numeric_features = contract["numeric_features"]
    categorical_features = contract["categorical_features"]
    target_col = contract["target_col"]

    # split до кодирования категорий (см. target_encode) — иначе утечка test->train.
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["price_segment"]
    )
    print(f"train: {len(df_train)} test: {len(df_test)}")

    X_train, X_test = build_design_matrix(df_train, df_test, numeric_features, categorical_features, target_col)
    y_train, y_test = df_train[target_col].values, df_test[target_col].values
    feature_names = X_train.columns.tolist()

    models = {}

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    ridge_grid = GridSearchCV(ridge, {"ridge__alpha": [0.1, 1.0, 10.0, 50.0]}, cv=5, scoring="neg_root_mean_squared_error")
    ridge_grid.fit(X_train, y_train)
    models["ridge"] = ridge_grid.best_estimator_
    print("Ridge best params:", ridge_grid.best_params_)

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    rf_grid = GridSearchCV(
        rf,
        {"n_estimators": [200], "max_depth": [12, 20, None], "min_samples_leaf": [1, 5]},
        cv=3,
        scoring="neg_root_mean_squared_error",
    )
    rf_grid.fit(X_train, y_train)
    models["random_forest"] = rf_grid.best_estimator_
    print("RandomForest best params:", rf_grid.best_params_)

    lgbm = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    lgbm_grid = GridSearchCV(
        lgbm,
        {
            "n_estimators": [300, 600],
            "num_leaves": [15, 31],
            "learning_rate": [0.05, 0.1],
        },
        cv=3,
        scoring="neg_root_mean_squared_error",
    )
    lgbm_grid.fit(X_train, y_train)
    models["lightgbm"] = lgbm_grid.best_estimator_
    print("LightGBM best params:", lgbm_grid.best_params_)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    predictions = df_test[["offer_id", "price_rub", "price_segment"]].copy().reset_index(drop=True)
    predictions["log_price_rub_actual"] = y_test
    for name, model in models.items():
        pred_log = model.predict(X_test)
        predictions[f"{name}_pred_log"] = pred_log
        predictions[f"{name}_pred_rub"] = np.expm1(pred_log)
    predictions.to_csv(PROCESSED_DIR / "predictions.csv", index=False)

    X_train.assign(**{target_col: y_train}).to_csv(PROCESSED_DIR / "model_input_train.csv", index=False)
    X_test.assign(**{target_col: y_test}).to_csv(PROCESSED_DIR / "model_input_test.csv", index=False)
    Path("output/tables/feature_names.json").write_text(json.dumps(feature_names, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved models -> {MODELS_DIR}, predictions -> {PROCESSED_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
