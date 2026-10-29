from pathlib import Path
import json, ast
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from features import extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_leakage_current.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    features, labels, sample_ids = [], [], []
    for row in df.itertuples(index=False):
        signal = np.asarray(json.loads(row.signal), dtype=float)
        features.append(extract_features(signal, int(row.sampling_rate_hz)))
        labels.append(row.label)
        sample_ids.append(row.sample_id)
    X = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.asarray(labels)
    # Split whole signal windows before model fitting; each row is one independent generated window.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    models = {
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                                  random_state=42, min_samples_leaf=2),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                                                                  l2_regularization=0.1, random_state=42)
    }
    summary = {"dataset": "synthetic simulation", "test_size": 0.25,
               "features": list(X.columns), "class_order": sorted(np.unique(y).tolist()), "models": {}}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        summary["models"][name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_test, pred, labels=summary["class_order"]).tolist()
        }
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"\n{name}: accuracy={summary['models'][name]['accuracy']:.3f}")
        print(classification_report(y_test, pred, zero_division=0))
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")
    with open(REPORT_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved models to {MODEL_DIR}")
    print(f"Saved evaluation metrics to {REPORT_DIR / 'evaluation_metrics.json'}")

if __name__ == "__main__":
    main()
