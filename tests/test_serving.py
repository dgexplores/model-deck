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


def test_provenance_and_training_only_monitor_reference(tmp_path):
    import hashlib
    import numpy as np
    from sklearn.datasets import load_wine
    from train import ROOT

    result = train(tmp_path)
    reference = load_wine().data[result['split']['train_indices']]
    assert result['schema_version'] == 2
    assert result['monitoring_reference']['rows'] == 106
    np.testing.assert_allclose(result['monitoring_reference']['feature_mean'], reference.mean(axis=0))
    for name, digest in result['provenance']['source_sha256'].items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_drift_detects_shift_but_not_identical_reference():
    import numpy as np
    from sklearn.datasets import load_wine
    from monitoring import drift_report

    dataset = load_wine()
    reference = dataset.data[:100]
    stable = drift_report(reference, reference, dataset.feature_names)
    assert stable['status'] == 'no_shift_detected'
    assert all(row['ks_statistic'] == 0 for row in stable['features'])
    shifted = reference.copy()
    shifted[:, 0] += 10 * np.std(reference[:, 0])
    result = drift_report(shifted, reference, dataset.feature_names)
    assert result['shifted_features'] == [dataset.feature_names[0]]
    assert result['features'][0]['corrected_p_value'] < result['family_alpha']
    assert result['features'][0]['outside_training_range'] == 100


def test_drift_contract_and_no_inference_side_effect(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    with TestClient(app) as client:
        features = client.get('/api/samples').json()[0]['features']
        assert client.post('/api/drift', json={'rows': [features] * 29}).status_code == 422
        assert client.post('/api/drift', json={'rows': [[1, 2]] * 30}).status_code == 422
        assert client.post('/api/drift', json={'rows': [features] * 1001}).status_code == 422
        result = client.post('/api/drift', json={'rows': [features] * 30})
        assert result.status_code == 200
        assert result.json()['reference_rows'] == 106
        assert len(result.json()['features']) == 13
        assert client.get('/api/metrics').json()['requests'] == 0


def csv_data(names, rows):
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(names)
    writer.writerows(rows)
    return output.getvalue()


def test_csv_reordered_quoted_columns_match_json_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    with TestClient(app) as client:
        names = client.get('/api/registry').json()['feature_names']
        row = client.get('/api/samples').json()[0]['features']
        expected = client.post('/api/predict', json={'rows': [row]}).json()['predictions']
        data = '\ufeff' + csv_data(names[::-1], [row[::-1]])
        response = client.post('/api/predict-csv', content=data.encode('utf-8'), headers={'Content-Type': 'text/csv'})
        assert response.status_code == 200
        assert response.json()['predictions'] == expected
        assert response.json()['input_rows'] == 1
        assert response.json()['feature_order'] == names
        assert client.post('/api/predict-csv?model_id=missing', content=data.encode('utf-8'), headers={'Content-Type': 'text/csv'}).status_code == 404


def test_csv_rejects_bad_uploads_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    with TestClient(app) as client:
        names = client.get('/api/registry').json()['feature_names']
        row = client.get('/api/samples').json()[0]['features']
        headers = {'Content-Type': 'text/csv'}
        bad = ['', csv_data(names, []), csv_data(names[:-1], [row[:-1]]), csv_data([names[0]] * 13, [row]), csv_data(names, [[1, 2]]), csv_data(names, [row] * 101), csv_data(names, [['nan'] * 13]), csv_data(names, [['inf'] * 13]), csv_data(names, [['=1+1'] * 13]), csv_data(names, [[100001] * 13])]
        for data in bad:
            assert client.post('/api/predict-csv', content=data, headers=headers).status_code == 422
        assert client.post('/api/predict-csv', content=b'\xff', headers=headers).status_code == 422
        assert client.post('/api/predict-csv', content='a' * 131073, headers=headers).status_code == 413
        assert client.post('/api/predict-csv', content='{}', headers={'Content-Type': 'application/json'}).status_code == 415
        assert client.get('/api/metrics').json()['requests'] == 0
        data = csv_data(names, [row] * 29)
        assert client.post('/api/drift-csv', content=data, headers=headers).status_code == 422
        data = csv_data(names, [row] * 30)
        assert client.post('/api/drift-csv', content=data, headers=headers).status_code == 200
