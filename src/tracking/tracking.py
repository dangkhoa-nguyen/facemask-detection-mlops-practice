from pathlib import Path
import argparse
import re
import yaml
import pandas as pd
import mlflow

EXPERIMENT_NAME = "FaceMaskDetection"

# ===================================
# Find the latest training directory
# ===================================
def get_latest_train_dir(report_dir: Path) -> Path:
    """
    Find the latest training directory.

    Example:
        train1
        train2
        train3

    -> return train3
    """
    train_dirs = [
        d for d in report_dir.iterdir()
        if d.is_dir() and d.name.startswith("train")
    ]

    if not train_dirs:
        raise FileNotFoundError(
            f"No train directories found in {report_dir}"
        )

    return max(
        train_dirs,
        key=lambda d: int(d.name.replace("train", ""))
    )

# ========================================
# Load training metadata and resolve paths
# ========================================
def load_training_info(family: str):
    """
    Load training metadata and resolve paths.

    Returns:
        train_args
        metrics_df
        train_dir
        model_path
        run_name
    """
    reports_root = Path("reports") / family

    train_dir = get_latest_train_dir(reports_root)

    args_yaml = train_dir / "args.yaml"
    results_csv = train_dir / "results.csv"

    if not args_yaml.exists():
        raise FileNotFoundError(args_yaml)

    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    with open(args_yaml, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)

    model_name = Path(train_args["model"]).stem

    run_name = f"{model_name}_{train_dir.name}"

    model_path = Path("models") / family / f"{model_name}.onnx"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    metrics_df = pd.read_csv(results_csv)

    return (
        train_args,
        metrics_df,
        train_dir,
        model_path,
        run_name,
    )

# ==============
# Log parameters
# ==============
def log_params(train_args: dict):
    """
    Log training arguments as MLflow parameters.
    """
    ignored_params = {
        "data",
        "project",
        "save_dir",
        "tracker"
    }

    for key, value in train_args.items():
        if key in ignored_params:
            continue
        if value is not None:
            mlflow.log_param(key, value)


# =============================================================
# Convert Ultralytics metric names into MLflow-compatible names
# =============================================================
def sanitize_metric_name(metric_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", metric_name)


# ===========
# Log metrics
# ===========
def log_metrics(metrics_df: pd.DataFrame):
    """
    Log metric history from results.csv.

    Each epoch becomes one MLflow step.
    """
    metric_columns = [
        col
        for col in metrics_df.columns
        if col not in ["epoch", "time"]
    ]

    for _, row in metrics_df.iterrows():
        step = int(row["epoch"])

        for metric in metric_columns:
            value = row[metric]

            if pd.notna(value):
                metric_name = sanitize_metric_name(metric)
                mlflow.log_metric(
                    metric_name,
                    float(value),
                    step=step,
                )

# =============
# Log artifacts
# =============
def log_artifacts(train_dir: Path, model_path: Path):
    """
    Log training outputs and exported model.

    Training directory:
        args.yaml
        results.csv
        plots
        confusion matrix
        ...

    Model:
        *.onnx
    """
    mlflow.log_artifacts(train_dir)

    mlflow.log_artifact(model_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--family",
        required=True,
        help="Model family (yolo26, yolo11, rtdetr, ...)",
    )

    args = parser.parse_args()

    (
        train_args,
        metrics_df,
        train_dir,
        model_path,
        run_name,
    ) = load_training_info(args.family)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):

        log_params(train_args)

        log_metrics(metrics_df)

        log_artifacts(train_dir, model_path)

    print(f"Logged run: {run_name}")


if __name__ == "__main__":
    main()