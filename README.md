# Model Deck

**Live demo:** [dg-model-deck.onrender.com](https://dg-model-deck.onrender.com) · [API reference](https://dg-model-deck.onrender.com/docs)

**An experiment-to-endpoint workbench: train, compare, version, and serve three real classifiers.**

Model Deck makes the delivery side of machine learning visible. A reviewer can inspect validation and test scores, choose a model version, edit a sample, and call the same inference endpoint used by the interface. No provider key is needed.

![Model workbench](docs/model-workbench.png)

## Why build this?

A notebook score does not show how a model becomes an application. This project connects training, selection, artifact identity, input validation, batch inference, and operational measurements in one reproducible workflow.

The example task classifies wine cultivars from thirteen chemical measurements using scikit-learn's UCI Wine dataset. It is a small educational benchmark, not a wine-quality estimator or an industrial deployment claim.

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the workbench or `/docs` for the interactive API reference. The first startup can train missing artifacts; deployment explicitly trains them during the build.

## What should a recruiter try?

1. Compare the registry cards. The validation champion is deliberately retained even when a different candidate scores higher on the held-out test.
2. Choose a held-out example and model version. Run a prediction and inspect probabilities and measured inference time.
3. Edit a measurement and repeat. Inspect the manifest to see feature order and exact split membership.
4. Send a malformed request through `/docs`; it should be rejected instead of producing a misleading prediction.

## Architecture

```mermaid
flowchart LR
  D[UCI Wine data] --> S[Stratified train / validation / test]
  S --> T[Fit candidate pipelines on training data]
  T --> V[Select using validation macro F1]
  V --> R[Refit on train plus validation]
  R --> E[Evaluate once on held-out test]
  E --> A[Artifacts and SHA-256 manifest]
  A --> API[FastAPI versioned inference]
  UI[Browser workbench] --> API
  API --> M[SQLite aggregate timing]
```

Models: standardized logistic regression, random forest, and a small neural network with two hidden layers (24 and 12 units). Preprocessing is fitted inside each pipeline on its fitting partition. The neural model demonstrates basic neural-network training; this does not imply LLM training experience.

## Measured results

The committed [manifest](artifacts/manifest.json) records the generated results, runtime versions, dataset fingerprint, split indices, and model digests. Run `python train.py` to reproduce them under the pinned dependencies. Floating-point results can vary across platforms.

On the initial local run (106 training, 36 validation, 36 test observations), neural-v1 won validation selection with macro F1 1.000 and achieved test accuracy 0.944. Linear-v1 achieved 0.972 test accuracy; forest-v1 achieved 1.000. The majority-class baseline achieved 0.389. These small-sample results have substantial uncertainty. We do not switch the champion based on its test result, and the 100% forest score is not evidence of perfect real-world performance.

## API

| Route | Purpose |
|---|---|
| `GET /healthz` | Readiness and loaded-model count |
| `GET /api/registry` | Model versions, scores, provenance, split indices |
| `GET /api/samples` | Disclosed held-out examples |
| `POST /api/predict` | Up to 100 rows, each with 13 finite measurements |
| `GET /api/metrics` | Latest 2000 calls; aggregate inference timing |

Example body: `{"model_id":"champion","rows":[[14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065]]}`. `model_id` can also be `linear-v1`, `forest-v1`, or `neural-v1`.

## Engineering decisions and limitations

- **Why a small dataset?** Training is quick, deterministic, and inspectable. Serving and release discipline are the focus. A larger independent dataset is needed for external validity.
- **Why no arbitrary model upload?** Python serialized artifacts can execute code. The service loads only project-generated artifacts checked against the build manifest. Hashes detect mismatch; they do not authenticate an attacker-controlled manifest. The build source remains trusted.
- **Why one worker?** This bounded demo uses local SQLite for aggregate telemetry. Multi-instance serving needs a shared metrics store, authentication, rate limiting and load testing.
- **What persists?** The free-hosting filesystem can be reset on deploy/restart. Metrics are demonstration telemetry, not a durable service history. Input feature rows are not stored.
- **What does latency mean?** Model inference inside the process, excluding network and cold-start latency. p95 is calculated from the latest retained calls, not a throughput benchmark.
- **What about AWS?** The architecture is inspired by containerized inference, but this project does not claim verified Lambda deployment. The supplied Docker and Render configuration are its supported deployment paths.

## Verification and deployment

```bash
pip install pytest httpx
python -m pytest -q
docker build -t model-deck .
docker run --rm -p 8000:8000 -e PORT=8000 model-deck
```

The tests cover disjoint partitions, validation-only selection, real prediction contracts, invalid versions and inputs, and aggregate telemetry. CI runs tests and builds the container. `render.yaml` describes a free Python service. A live URL should be added only after successful remote deployment and smoke checks.

## Ownership, inspiration, and skills

An original implementation for Deepak Gangwar's portfolio, developed with AI assistance. [SeamlessML](https://github.com/newtonjain/Docker_Hackathon) inspired the model-packaging and serving problem; no upstream code, model weights, notebooks, or UI were copied. Its original authors' award is not this project's achievement. See [reference analysis](docs/reference-analysis.md).

Demonstrated skills: Python, REST APIs, software design, Machine Learning, a small feedforward neural network, model selection, data processing, MLOps foundations, Git/CI, testing, and technical documentation. No enterprise users, business savings, or production reliability figures are claimed.

Dataset attribution: UCI Wine recognition, distributed through scikit-learn; see `sklearn.datasets.load_wine().DESCR` for the original dataset description and references. Source code: MIT [license](LICENSE).

## Original project study

See the [upstream project study](docs/upstream-study.md) for the original architecture, authorship, setup requirements and limitations. It describes a separate upstream reference, not this repository’s deployed application.
