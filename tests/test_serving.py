import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from app import app
from train import train


def test_training_has_disjoint_splits_and_artifact_identity(tmp_path):
    result = train(tmp_path)
    split = result["split"]
    groups = [set(split[key]) for key in ("train_indices","validation_indices","test_indices")]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert len(set.union(*groups)) == 178
    assert result["champion"] == max(result["models"],key=lambda key: result["models"][key]["validation_macro_f1"])
    assert result["models"][result["champion"]]["test_accuracy"] > result["baseline_test_accuracy"]
    assert all((tmp_path/f"{name}.joblib").exists() for name in result["models"])


def test_prediction_versions_shape_and_invalid_input(tmp_path,monkeypatch):
    monkeypatch.setenv("DATA_DIR",str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/").status_code==200
        assert client.get("/static/app.js").status_code==200
        samples=client.get("/api/samples").json()
        for name in client.get("/api/registry").json()["models"]:
            response=client.post("/api/predict",json={"model_id":name,"rows":[sample["features"] for sample in samples]})
            assert response.status_code==200
            assert len(response.json()["predictions"])==3
            assert all(abs(sum(p["probabilities"].values())-1)<.0001 for p in response.json()["predictions"])
        assert client.post("/api/predict",json={"rows":[[1,2]]}).status_code==422
        assert client.post("/api/predict",json={"rows":[[1]*13]*101}).status_code==422
        assert client.post("/api/predict",json={"model_id":"missing","rows":[[1]*13]}).status_code==404
        assert client.get("/api/metrics").json()["requests"]==3
