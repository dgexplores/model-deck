"""Reproducible, leakage-aware model registry. No user-provided model loading."""
from pathlib import Path
import hashlib
import json
import platform

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def train(output=ARTIFACTS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_wine()
    indices = np.arange(len(dataset.target))
    fit_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=dataset.target, random_state=42)
    train_idx, val_idx = train_test_split(fit_idx, test_size=0.25, stratify=dataset.target[fit_idx], random_state=43)
    candidates = {
        "linear-v1": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "forest-v1": RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42, n_jobs=1),
        "neural-v1": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(24, 12), solver="lbfgs", max_iter=800, random_state=42)),
    }
    reports = {}
    # Selection uses validation only. The held-out test never chooses a model.
    for name, estimator in candidates.items():
        estimator.fit(dataset.data[train_idx], dataset.target[train_idx])
        prediction = estimator.predict(dataset.data[val_idx])
        reports[name] = {"validation_macro_f1": float(f1_score(dataset.target[val_idx], prediction, average="macro"))}
    champion = max(candidates, key=lambda name: reports[name]["validation_macro_f1"])
    for name, estimator in candidates.items():
        estimator.fit(dataset.data[fit_idx], dataset.target[fit_idx])
        prediction = estimator.predict(dataset.data[test_idx])
        path = output / f"{name}.joblib"
        joblib.dump(estimator, path)
        reports[name].update({
            "test_accuracy": float(accuracy_score(dataset.target[test_idx], prediction)),
            "test_macro_f1": float(f1_score(dataset.target[test_idx], prediction, average="macro")),
            "confusion_matrix": confusion_matrix(dataset.target[test_idx], prediction).tolist(),
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "artifact_bytes": path.stat().st_size,
            "algorithm": type(estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator).__name__,
        })
    baseline = DummyClassifier(strategy="most_frequent").fit(dataset.data[fit_idx], dataset.target[fit_idx])
    manifest = {
        "schema_version": 1, "dataset": "UCI Wine recognition via sklearn.datasets.load_wine (178 observations, 3 cultivars)",
        "dataset_sha256": hashlib.sha256(dataset.data.tobytes()+dataset.target.tobytes()).hexdigest(),
        "feature_names": dataset.feature_names, "classes": dataset.target_names.tolist(),
        "champion": champion, "models": reports,
        "split": {"seed_test":42,"seed_validation":43,"train_indices":train_idx.tolist(),"validation_indices":val_idx.tolist(),"test_indices":test_idx.tolist()},
        "baseline_test_accuracy": float(baseline.score(dataset.data[test_idx], dataset.target[test_idx])),
        "runtime": {"python":platform.python_version(),"sklearn":sklearn.__version__,"numpy":np.__version__},
        "limitations": "Small educational dataset; held-out results are not production generalization evidence. Neural-v1 is a small two-hidden-layer network, not an LLM.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    result = train()
    print(json.dumps({"champion":result["champion"],"models":result["models"],"baseline_test_accuracy":result["baseline_test_accuracy"]},indent=2))
