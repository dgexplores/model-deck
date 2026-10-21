# SeamlessML reference study

Inspected source: `newtonjain/Docker_Hackathon`, commit `d6db83f2a0307dfe82da7dfa46891d79f15fb1ac`, retrieved 2026-10-22.

## Observed source facts

- `Scikit-learn Machine Learning Inference API/app/app.py`: loads a bundled joblib artifact from `/opt/ml/model`, base64-decodes and resizes an image, deskews it using image moments, then calls `model.predict` from a Lambda handler.
- `Scikit-learn Machine Learning Inference API/template.yaml`: SAM image-packaged function, x86_64 architecture, API Gateway POST `/classify_digit`, timeout 50 seconds and 5000 MB memory configuration.
- The tree contains separate framework examples for scikit-learn, XGBoost, TensorFlow, and PyTorch.
- The public project description identifies model deployment, local testing, container packaging and cold-start trade-offs as its focus.

## Boundaries

This was a bounded static source inspection, not execution of its Lambda deployment, notebooks or serialized model artifacts. Its latency, cost and scale were not verified. No repository license was exposed by GitHub metadata at inspection, so no upstream source was copied into this implementation.

## Independent design choices

Model Deck uses an original tabular classification workflow, split manifests, validation-only champion selection, explicit model versions, SHA-256 artifact checks, typed FastAPI contracts, an interactive comparison interface, and SQLite timing. It does not reconstruct the winner's digit-image pipeline or claim feature parity with every framework example. The public API deliberately omits arbitrary model upload and training execution.

The REA skill routes complete source repositories to normal source analysis; no binary decompiler or REA runtime measurement was used or claimed.
