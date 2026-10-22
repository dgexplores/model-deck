# Demonstrate and explain this project

1. Open the model registry. Explain the106/36/36 training/validation/test split and why the validation champion is not reselected using test results.
2. Run a held-out example. Inspect the actual probabilities, model version and inference latency. Probabilities are not calibrated guarantees.
3. Download the held-out CSV example and upload it. Run CSV prediction and inspect all three rows.
4. Drift checks need at least30 rows; a three-row example is intentionally insufficient. Use a valid larger CSV, inspect feature-level KS evidence and corrected p-values.
5. Inspect the manifest for artifact and source identities. Retrain with train.py when changing the model code.

Explain: preprocessing leakage, baseline classifier, macroF1, drift versus accuracy degradation, multiple-testing correction and source/artifact provenance. The synthetic80-case drift experiment is not a production false-positive guarantee.
