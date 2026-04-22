import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# -----------------------------
# AeroPredict MVP (Phase 3)
# NASA C-MAPSS FD001 baseline
# -----------------------------

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Change this if your data is somewhere else
DATA_DIR = Path(__file__).parent / "data"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"


def load_cmapss_file(filepath: Path) -> pd.DataFrame:
    """
    Loads NASA C-MAPSS txt files.
    Expected format:
    unit_nr, time_cycles, op_setting_1, op_setting_2, op_setting_3, sensor_1 ... sensor_21
    """
    column_names = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df = df.iloc[:, :26]  # remove any extra blank columns if present
    df.columns = column_names
    return df


def add_rul_to_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    RUL = max cycle for each engine - current cycle
    """
    max_cycles = train_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="engine_id", how="left")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df = train_df.drop(columns=["max_cycle"])
    return train_df


def build_test_targets(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Official test RUL file gives remaining cycles AFTER the last observed test cycle.
    True RUL at each test row = final_test_rul + (last_cycle_for_engine - current_cycle)
    """
    last_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    last_cycles.columns = ["engine_id", "last_cycle"]

    rul_df = rul_df.copy()
    rul_df.columns = ["final_rul"]

    # C-MAPSS RUL rows are in engine order starting at 1
    rul_df["engine_id"] = np.arange(1, len(rul_df) + 1)

    test_targets = last_cycles.merge(rul_df, on="engine_id", how="left")
    test_targets["true_RUL_last_cycle"] = test_targets["final_rul"]

    return test_targets


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Drop near-constant sensors later if needed.
    For now use operating settings + all sensors except identifiers/target.
    """
    excluded = {"engine_id", "cycle", "RUL"}
    return [col for col in df.columns if col not in excluded]


def drop_constant_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, threshold: float = 1e-8):
    """
    Removes columns with almost no variance in training data.
    """
    keep_cols = []
    for col in feature_cols:
        if train_df[col].var() > threshold:
            keep_cols.append(col)

    return keep_cols


def get_last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets one row per engine: the most recent cycle.
    This is the row used for final engine-level RUL prediction.
    """
    idx = df.groupby("engine_id")["cycle"].idxmax()
    return df.loc[idx].sort_values("engine_id").reset_index(drop=True)


def bootstrap_prediction_intervals(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_bootstrap: int = 10,
    n_estimators: int = 150,
):
    """
    Bootstrap intervals by resampling training rows with replacement
    and retraining the model.
    """
    preds = []

    for i in range(n_bootstrap):
        sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_sample = X_train.iloc[sample_idx]
        y_sample = y_train.iloc[sample_idx]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE + i,
            n_jobs=-1,
        )
        model.fit(X_sample, y_sample)
        pred = model.predict(X_test)
        preds.append(pred)

    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)

    return mean_pred, lower, upper


def save_results(engine_ids, true_rul, pred_rul, lower, upper, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "engine_id": engine_ids,
        "true_RUL": true_rul,
        "predicted_RUL": np.round(pred_rul, 2),
        "lower_bound_90pct": np.round(lower, 2),
        "upper_bound_90pct": np.round(upper, 2),
    })

    results.to_csv(output_dir / "mvp_predictions.csv", index=False)
    return results


def make_plot(results_df: pd.DataFrame, output_dir: Path):
    plt.figure(figsize=(10, 6))
    x = results_df["engine_id"]
    y = results_df["predicted_RUL"]
    y_true = results_df["true_RUL"]
    yerr_lower = y - results_df["lower_bound_90pct"]
    yerr_upper = results_df["upper_bound_90pct"] - y

    plt.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=4, label="Predicted RUL with 90% interval")
    plt.plot(x, y_true, "x", label="True RUL")
    plt.xlabel("Engine ID")
    plt.ylabel("Remaining Useful Life (cycles)")
    plt.title("AeroPredict MVP: Test Engine RUL Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rul_predictions_plot.png", dpi=300)
    plt.close()


def main():
    print("AeroPredict MVP (Phase 3)")
    print("-" * 40)

    # Check files
    missing_files = [str(f) for f in [TRAIN_FILE, TEST_FILE, RUL_FILE] if not f.exists()]
    if missing_files:
        print("ERROR: Missing required dataset files.")
        print("Put these files inside a folder named 'data' next to this script:")
        for f in missing_files:
            print(f" - {f}")
        print("\nExpected files:")
        print(" - data/train_FD001.txt")
        print(" - data/test_FD001.txt")
        print(" - data/RUL_FD001.txt")
        return

    # Load data
    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)
    rul_df = pd.read_csv(RUL_FILE, sep=r"\s+", header=None)

    # Add train labels
    train_df = add_rul_to_training_data(train_df)

    # Build test true labels for last cycle of each engine
    test_targets = build_test_targets(test_df, rul_df)

    # Feature selection
    feature_cols = get_feature_columns(train_df)
    feature_cols = drop_constant_columns(train_df, test_df, feature_cols)

    # Train on all training rows
    X_train = train_df[feature_cols]
    y_train = train_df["RUL"]

    # Predict on only the last cycle row of each test engine
    test_last = get_last_cycle_rows(test_df)
    test_last = test_last.merge(
        test_targets[["engine_id", "true_RUL_last_cycle"]],
        on="engine_id",
        how="left"
    )

    X_test = test_last[feature_cols]
    y_test = test_last["true_RUL_last_cycle"]

    # Bootstrap ensemble prediction intervals
    mean_pred, lower_pred, upper_pred = bootstrap_prediction_intervals(
        X_train,
        y_train,
        X_test,
        n_bootstrap=50,
        n_estimators=150,
    )

    # Metrics
    mae = mean_absolute_error(y_test, mean_pred)
    rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
    coverage = np.mean((y_test >= lower_pred) & (y_test <= upper_pred))

    # Save outputs
    output_dir = Path("outputs")
    results_df = save_results(
        engine_ids=test_last["engine_id"],
        true_rul=y_test,
        pred_rul=mean_pred,
        lower=lower_pred,
        upper=upper_pred,
        output_dir=output_dir,
    )
    make_plot(results_df, output_dir)

    # Print summary
    print("Model evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"90% interval coverage: {coverage:.2%}")

    print("\nExample predictions:")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(
            f"Engine {int(row['engine_id'])}: "
            f"Predicted RUL = {row['predicted_RUL']:.2f} cycles, "
            f"Interval = [{row['lower_bound_90pct']:.2f}, {row['upper_bound_90pct']:.2f}], "
            f"True RUL = {row['true_RUL']:.2f}"
        )

    print("\nSaved files:")
    print(" - outputs/mvp_predictions.csv")
    print(" - outputs/rul_predictions_plot.png")


if __name__ == "__main__":
    main()
