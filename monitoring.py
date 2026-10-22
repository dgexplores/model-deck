"""Compare a batch with the fixed training partition without storing inputs."""
import numpy as np
from scipy.stats import ks_2samp


def drift_report(rows, reference, names):
    """Measure marginal feature shift using corrected two-sample KS tests."""
    incoming = np.asarray(rows, dtype=float)
    baseline = np.asarray(reference, dtype=float)
    family_alpha = 0.05
    cutoff = family_alpha / len(names)
    features = []
    for index, name in enumerate(names):
        expected, observed = baseline[:, index], incoming[:, index]
        statistic, p_value = ks_2samp(expected, observed, method="asymp")
        scale = float(np.std(expected, ddof=1))
        features.append({
            "feature": name,
            "ks_statistic": round(float(statistic), 6),
            "p_value": float(p_value),
            "corrected_p_value": min(1.0, float(p_value) * len(names)),
            "shift_detected": bool(p_value < cutoff),
            "mean_shift_training_std": round(float((observed.mean() - expected.mean()) / scale), 4) if scale else None,
            "outside_training_range": int(((observed < expected.min()) | (observed > expected.max())).sum()),
        })
    shifted = [row["feature"] for row in features if row["shift_detected"]]
    return {
        "status": "shift_detected" if shifted else "no_shift_detected",
        "rows": len(rows), "reference_rows": len(reference), "shifted_features": shifted,
        "family_alpha": family_alpha, "per_feature_alpha": cutoff,
        "method": "Two-sided two-sample Kolmogorov–Smirnov, asymptotic p-values; Bonferroni across 13 marginal features",
        "reference_scope": "Fixed original training partition only; validation and test rows excluded",
        "features": features,
        "limitations": "A monitoring signal, not proof of model failure or absence of drift. Assumes independent observations. Small batches have low power; joint-only changes, label shift and concept drift can be missed. No labels are inspected.",
    }
