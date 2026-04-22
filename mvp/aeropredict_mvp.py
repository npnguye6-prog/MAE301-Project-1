from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"


def load_cmapss_file(filepath: Path) -> pd.DataFrame:
    column_names = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = column_names
    return df


def add_rul_to_training_data(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = train_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    train_df = train_df.merge(max_cycles, on="engine_id", how="left")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    return train_df.drop(columns=["max_cycle"])


def build_test_targets(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    last_cycles = test_df.groupby("engine_id")["cycle"].max().reset_index()
    last_cycles.columns = ["engine_id", "last_cycle"]

    rul_df = rul_df.copy()
    rul_df.columns = ["final_rul"]
    rul_df["engine_id"] = np.arange(1, len(rul_df) + 1)

    out = last_cycles.merge(rul_df, on="engine_id", how="left")
    out["true_RUL_last_cycle"] = out["final_rul"]
    return out


def get_last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("engine_id")["cycle"].idxmax()
    return df.loc[idx].sort_values("engine_id").reset_index(drop=True)


def prepare_engine_level_features(df: pd.DataFrame) -> pd.DataFrame:
    last_rows = get_last_cycle_rows(df)
    keep_cols = (
        ["engine_id", "cycle"]
        + [f"op_setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    return last_rows[keep_cols].copy()


def main():
    print("AeroPredict MVP (Phase 3)")
    print("-" * 40)

    for f in [TRAIN_FILE, TEST_FILE, RUL_FILE]:
        if not f.exists():
            print(f"Missing file: {f}")
            return

    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)
    rul_df = pd.read_csv(RUL_FILE, sep=r"\s+", header=None)

    train_df = add_rul_to_training_data(train_df)

    train_last = prepare_engine_level_features(train_df)
    test_last = prepare_engine_level_features(test_df)

    train_targets = train_df.groupby("engine_id")["RUL"].min().reset_index()
    train_targets.columns = ["engine_id", "true_RUL_last_cycle"]

    test_targets = build_test_targets(test_df, rul_df)

    train_last = train_last.merge(train_targets, on="engine_id", how="left")
    test_last = test_last.merge(test_targets[["engine_id", "true_RUL_last_cycle"]], on="engine_id", how="left")

    feature_cols = [c for c in train_last.columns if c not in ["engine_id", "true_RUL_last_cycle"]]

    X_train = train_last[feature_cols]
    y_train = train_last["true_RUL_last_cycle"]
    X_test = test_last[feature_cols]
    y_test = test_last["true_RUL_last_cycle"]

    model = RandomForestRegressor(
        n_estimators=25,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # simple uncertainty estimate using tree spread
    tree_preds = np.array([tree.predict(X_test) for tree in model.estimators_])
    lower = np.percentile(tree_preds, 10, axis=0)
    upper = np.percentile(tree_preds, 90, axis=0)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    coverage = np.mean((y_test >= lower) & (y_test <= upper))

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"80% interval coverage: {coverage:.2%}")

    print("\nExample predictions:")
    for i in range(min(5, len(test_last))):
        print(
            f"Engine {int(test_last.loc[i, 'engine_id'])}: "
            f"Predicted RUL = {pred[i]:.2f} cycles, "
            f"Interval = [{lower[i]:.2f}, {upper[i]:.2f}], "
            f"True RUL = {y_test.iloc[i]:.2f}"
        )


if __name__ == "__main__":
    main()
