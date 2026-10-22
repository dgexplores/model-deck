"""A bounded model-serving workbench with measured, versioned predictions."""
from contextlib import asynccontextmanager
import csv
import io
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sklearn.datasets import load_wine

from train import ARTIFACTS, ROOT, train
from monitoring import drift_report


@asynccontextmanager
async def lifespan(application):
    if not (ARTIFACTS / "manifest.json").exists():
        train()
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    models = {}
    for name, report in manifest["models"].items():
        path = ARTIFACTS / f"{name}.joblib"
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["artifact_sha256"]:
            raise RuntimeError("Model artifact identity does not match its manifest")
        # Only this project's generated, hash-checked build artifacts are loaded.
        models[name] = joblib.load(path)
    application.state.manifest, application.state.models = manifest, models
    application.state.reference = load_wine().data[manifest["split"]["train_indices"]]
    data_dir = Path(os.getenv("DATA_DIR", "/tmp/model-deck"))
    data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(data_dir / "metrics.sqlite", check_same_thread=False)
    connection.execute("CREATE TABLE IF NOT EXISTS calls(model TEXT, rows INTEGER, elapsed_ms REAL, created REAL)")
    connection.commit()
    application.state.db, application.state.lock = connection, threading.Lock()
    yield
    connection.close()


app = FastAPI(title="Model Deck", description="Versioned inference with reproducible model selection and measured latency.", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.middleware("http")
async def bound_request_body(request: Request, call_next):
    if request.method == "POST":
        limit = 131072 if request.url.path.endswith("-csv") else 524288
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                return JSONResponse({"detail": "Request body exceeds the allowed size."}, status_code=413)
            body.extend(chunk)
        request._body = bytes(body)
    return await call_next(request)



class PredictionRequest(BaseModel):
    model_id: str = Field(default="champion", max_length=40)
    rows: list[list[float]] = Field(min_length=1, max_length=100)

    @field_validator("rows")
    @classmethod
    def validate_rows(cls, rows):
        if any(len(row) != 13 or any(not math.isfinite(value) or abs(value)>100_000 for value in row) for row in rows):
            raise ValueError("Each sample requires 13 finite feature values, each within ±100000")
        return rows


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/healthz")
def health():
    return {"status":"ok", "models":len(app.state.models), "mode":"educational-model-serving"}


@app.get("/api/registry")
def registry():
    return app.state.manifest


@app.get("/api/samples")
def samples():
    data = load_wine()
    # Examples come from the held-out set; labels are disclosed for inspection.
    indices = app.state.manifest["split"]["test_indices"]
    chosen = [next(i for i in indices if int(data.target[i])==label) for label in range(3)]
    return [{"name": f"Cultivar {int(data.target[i])+1}", "expected_class":str(data.target_names[data.target[i]]),"features":data.data[i].tolist()} for i in chosen]


@app.post("/api/predict")
def predict(payload: PredictionRequest):
    manifest = app.state.manifest
    name = manifest["champion"] if payload.model_id == "champion" else payload.model_id
    if name not in app.state.models:
        raise HTTPException(404,"Model version not found. Choose a version from the registry.")
    started = time.perf_counter()
    probabilities = app.state.models[name].predict_proba(payload.rows)
    elapsed = (time.perf_counter()-started)*1000
    with app.state.lock:
        app.state.db.execute("INSERT INTO calls VALUES(?,?,?,?)",(name,len(payload.rows),elapsed,time.time()))
        # Bounded demo history, never raw feature inputs or user identifiers.
        app.state.db.execute("DELETE FROM calls WHERE rowid NOT IN (SELECT rowid FROM calls ORDER BY rowid DESC LIMIT 2000)")
        app.state.db.commit()
    return {"model_id":name,"elapsed_ms":round(elapsed,3),"predictions":[{"class":manifest["classes"][int(row.argmax())],"probabilities":{label:round(float(value),5) for label,value in zip(manifest["classes"],row)}} for row in probabilities],"note":"Probabilities are model outputs, not calibrated guarantees."}


@app.get("/api/metrics")
def metrics():
    with app.state.lock:
        rows = app.state.db.execute("SELECT model, rows, elapsed_ms FROM calls ORDER BY rowid DESC LIMIT 2000").fetchall()
    latency = sorted(row[2] for row in rows)
    return {"requests":len(rows),"samples":sum(row[1] for row in rows),"latency_ms_p95":round(latency[max(0,math.ceil(len(latency)*.95)-1)],3) if latency else None,"scope":"Latest 2000 requests in this instance; free hosting storage is ephemeral. Inference time excludes network latency."}


class DriftRequest(PredictionRequest):
    rows: list[list[float]] = Field(min_length=30, max_length=1000)


@app.post("/api/drift")
def monitor(payload: DriftRequest):
    """Compare at least 30 rows against the fixed training reference."""
    return drift_report(payload.rows, app.state.reference, app.state.manifest["feature_names"])


async def parse_csv_request(request: Request, model_id: str = "champion"):
    """Validate a small UTF-8 CSV upload before invoking batch inference."""
    if request.headers.get("content-type", "").split(";")[0].strip() not in ("text/csv", "text/plain"):
        raise HTTPException(415, "Send UTF-8 CSV with Content-Type text/csv or text/plain.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 131072:
            raise HTTPException(413, "CSV exceeds the 128 KiB upload limit.")
        body.extend(chunk)
    try:
        text = body.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        expected = app.state.manifest["feature_names"]
        if len(header) != 13 or len(set(header)) != 13 or set(header) != set(expected):
            raise HTTPException(422, "CSV header must contain the 13 registry feature names exactly once. Column order may vary.")
        order = [header.index(name) for name in expected]
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != 13:
                raise HTTPException(422, "Every CSV record must contain exactly 13 measurements.")
            rows.append([float(row[index]) for index in order])
            if len(rows) > 100:
                raise HTTPException(422, "CSV accepts at most 100 data records.")
        payload = PredictionRequest(model_id=model_id, rows=rows)
    except (UnicodeError, csv.Error, StopIteration, ValueError) as error:
        raise HTTPException(422, "Invalid CSV. Supply a header and 1–100 rows of finite numeric measurements.") from error
    return payload


@app.post("/api/predict-csv")
async def predict_csv(request: Request, model_id: str = "champion"):
    payload = await parse_csv_request(request, model_id)
    result = predict(payload)
    return {**result, "input_rows": len(payload.rows), "feature_order": app.state.manifest["feature_names"]}


@app.post("/api/drift-csv")
async def monitor_csv(request: Request):
    payload = await parse_csv_request(request)
    if len(payload.rows) < 30:
        raise HTTPException(422, "Drift checks require at least 30 measurements.")
    return drift_report(payload.rows, app.state.reference, app.state.manifest["feature_names"])
