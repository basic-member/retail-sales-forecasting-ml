import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    assert "target" in df.columns, "Dataset must contain a 'target' column"
    assert df.shape[0] > 0, "Dataset is empty"

    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    # ======================
    # Load data
    # ======================
    df = load_data("sample_data.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    # ======================
    # Train / Test split
    # ======================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # ======================
    # Baseline (Mean Predictor)
    # ======================
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_metrics = evaluate(y_test, baseline_pred)

    # ======================
    # Model pipeline
    # ======================
    pipeline = build_pipeline()

    # ======================
    # Hyperparameter tuning (lightweight by design)
    # ======================
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 15],
        "model__min_samples_split": [2, 5],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # ======================
    # Evaluation
    # ======================
    preds = best_model.predict(X_test)
    model_metrics = evaluate(y_test, preds)

    # ======================
    # Reporting
    # ======================
    print("Final Model Results")
    print("-------------------")
    print(f"MAE : {model_metrics['MAE']:.2f}")
    print(f"RMSE: {model_metrics['RMSE']:.2f}")
    print(f"R2  : {model_metrics['R2']:.3f}")

    print("\nBaseline (Mean Predictor)")
    print("------------------------")
    print(f"MAE : {baseline_metrics['MAE']:.2f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.2f}")
    print(f"R2  : {baseline_metrics['R2']:.3f}")

    print("\nBest Params:", grid.best_params_)


if __name__ == "__main__":
    main()
