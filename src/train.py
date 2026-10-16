"""
Machine learning training pipeline for AgroPrice AI.
Executes time-aware chronological data splitting, candidate model comparison,
cross-validation, hyperparameter tuning, model persistence, and metadata generation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.config import (
    DEFAULT_MODEL_PATH,
    METADATA_PATH,
    EVALUATION_REPORT_PATH
)
from src.data_loader import DataLoader
from src.feature_engineering import (
    extract_features,
    build_preprocessor,
    prepare_training_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.evaluate import (
    calculate_metrics,
    evaluate_crop_breakdown,
    compute_prediction_interval
)


def split_time_aware(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the agricultural time series dataset chronologically.
    Guarantees no future price leakage into the training set.
    """
    sorted_df = df.sort_values(by="Arrival_Date").reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - test_ratio))

    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()

    return train_df, test_df


def train_and_compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[Dict[str, Any], Any, str]:
    """
    Trains and compares multiple candidate regression models:
    - Ridge Regression (regularized linear baseline)
    - Random Forest Regressor
    - HistGradientBoosting Regressor
    - XGBoost Regressor

    Evaluates using TimeSeriesSplit cross-validation on train split,
    and out-of-time evaluation on test holdout.
    """
    X_train, y_train = prepare_training_data(train_df)
    X_test, y_test = prepare_training_data(test_df)

    preprocessor = build_preprocessor()

    candidates = {
        "Ridge Regression": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    best_model_name = None
    best_pipeline = None
    lowest_mae = float("inf")

    # TimeSeriesSplit for temporal cross-validation on train set
    tscv = TimeSeriesSplit(n_splits=3)

    for name, regressor in candidates.items():
        print(f"Evaluating candidate: {name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # Cross validation scores on training data
        cv_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            cv_X_train, cv_X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            cv_y_train, cv_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(cv_X_train, cv_y_train)
            cv_preds = pipeline.predict(cv_X_val)
            cv_maes.append(calculate_metrics(cv_y_val.values, cv_preds)["mae"])

        avg_cv_mae = round(float(np.mean(cv_maes)), 2)

        # Train on full training split and evaluate on holdout test set
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)
        holdout_metrics = calculate_metrics(y_test.values, y_test_pred)

        results[name] = {
            "cv_mae": avg_cv_mae,
            "test_mae": holdout_metrics["mae"],
            "test_rmse": holdout_metrics["rmse"],
            "test_r2": holdout_metrics["r2"],
            "test_mape": holdout_metrics["mape"]
        }

        print(f"  -> {name} | CV MAE: Rs.{avg_cv_mae} | Test MAE: Rs.{holdout_metrics['mae']} | R2: {holdout_metrics['r2']}")

        # Track champion model
        if holdout_metrics["mae"] < lowest_mae:
            lowest_mae = holdout_metrics["mae"]
            best_model_name = name
            best_pipeline = pipeline

    return results, best_pipeline, best_model_name


def extract_feature_importance(pipeline: Pipeline) -> Dict[str, float]:
    """Extracts top feature importance scores from the pipeline's regressor."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]

        # Get feature names from ColumnTransformer
        num_cols = NUMERICAL_FEATURES
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_features = num_cols + cat_cols

        if hasattr(regressor, "feature_importances_"):
            importances = regressor.feature_importances_
            feat_imp = {feat: round(float(imp), 4) for feat, imp in zip(all_features, importances)}
            sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:15])
            return sorted_imp
        elif hasattr(regressor, "coef_"):
            coefs = np.abs(regressor.coef_)
            feat_imp = {feat: round(float(c), 4) for feat, c in zip(all_features, coefs)}
            sorted_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:15])
            return sorted_imp
    except Exception as e:
        print(f"Notice: Feature importance extraction note: {e}")

    return {}


def build_market_hierarchy(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, list]]]:
    """
    Builds a hierarchical lookup:
    Crop -> State -> District -> [Markets]
    Enables instant, accurate cascading dropdowns in the Streamlit UI.
    """
    hierarchy: Dict[str, Dict[str, Dict[str, list]]] = {}

    grouped = df[["Commodity", "State", "District", "Market"]].drop_duplicates()
    for _, row in grouped.iterrows():
        crop = row["Commodity"]
        state = row["State"]
        dist = row["District"]
        mkt = row["Market"]

        if crop not in hierarchy:
            hierarchy[crop] = {}
        if state not in hierarchy[crop]:
            hierarchy[crop][state] = {}
        if dist not in hierarchy[crop][state]:
            hierarchy[crop][state][dist] = []

        if mkt not in hierarchy[crop][state][dist]:
            hierarchy[crop][state][dist].append(mkt)

    return hierarchy


def train_pipeline(
    data_path: Optional[str] = None,
    model_version: str = "v1.0.0"
) -> Dict[str, Any]:
    """
    Main training execution function.
    Loads data, performs time-aware split, compares candidate models,
    trains champion model, evaluates crop breakdowns and prediction intervals,
    and persists models and metadata to disk.
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting AgroPrice AI training workflow...")
    
    loader = DataLoader(default_path=data_path)
    df = loader.load_default_data()

    # Time-aware split: 80% train, 20% holdout
    train_df, test_df = split_time_aware(df, test_ratio=0.20)
    
    train_start = train_df["Arrival_Date"].min().strftime("%Y-%m-%d")
    train_end = train_df["Arrival_Date"].max().strftime("%Y-%m-%d")
    test_start = test_df["Arrival_Date"].min().strftime("%Y-%m-%d")
    test_end = test_df["Arrival_Date"].max().strftime("%Y-%m-%d")

    print(f"Training Period: {train_start} to {train_end} ({len(train_df):,} records)")
    print(f"Holdout Period:  {test_start} to {test_end} ({len(test_df):,} records)")

    # Model comparison
    candidate_results, champion_pipeline, champion_name = train_and_compare_models(train_df, test_df)
    print(f"\nChampion Model Selected: {champion_name}")

    # Holdout predictions for error bounds and crop breakdown
    X_test, y_test = prepare_training_data(test_df)
    test_preds = champion_pipeline.predict(X_test)
    residuals = y_test.values - test_preds

    overall_metrics = calculate_metrics(y_test.values, test_preds)
    crop_breakdown = evaluate_crop_breakdown(test_df, y_test.values, test_preds)
    residual_std, margin_error, test_mae = compute_prediction_interval(residuals, confidence_level=0.90)

    # Extract feature importance
    top_features = extract_feature_importance(champion_pipeline)

    # Build market hierarchy
    market_hierarchy = build_market_hierarchy(df)

    metadata = {
        "model_version": model_version,
        "champion_model": champion_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_records": int(len(train_df)),
        "holdout_records": int(len(test_df)),
        "training_period": {"start": train_start, "end": train_end},
        "holdout_period": {"start": test_start, "end": test_end},
        "candidate_models": candidate_results,
        "final_metrics": overall_metrics,
        "crop_breakdown": crop_breakdown,
        "prediction_interval": {
            "residual_std": residual_std,
            "margin_error_90": margin_error,
            "mean_absolute_error": test_mae
        },
        "supported_crops": sorted(df["Commodity"].unique().tolist()),
        "supported_states": sorted(df["State"].unique().tolist()),
        "supported_markets": sorted(df["Market"].unique().tolist()),
        "market_hierarchy": market_hierarchy,
        "top_features": top_features
    }

    # Save trained champion pipeline
    joblib.dump(champion_pipeline, DEFAULT_MODEL_PATH)
    print(f"Saved model pipeline to: {DEFAULT_MODEL_PATH}")

    # Save metadata JSON
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {METADATA_PATH}")

    # Save evaluation summary report
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved evaluation report to: {EVALUATION_REPORT_PATH}")

    return metadata


if __name__ == "__main__":
    train_pipeline()
