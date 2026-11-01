"""End-to-end checks for the saved notebook model and Flask form."""
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import FIELDS, METADATA_PATH, MODEL_PATH, create_app, validate_input  # noqa: E402


VALID_INPUT = {
    "age": "54", "sex": "1", "cp": "4", "trestbps": "130", "chol": "246",
    "fbs": "0", "restecg": "2", "thalach": "150", "exang": "0",
    "oldpeak": "1.2", "slope": "2", "ca": "1", "thal": "7",
}


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    for field in FIELDS:
        assert f'name="{field["name"]}"'.encode() in response.data


def test_about_page_loads(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"official UCI Heart Disease dataset" in response.data


def test_valid_prediction_and_output(client):
    response = client.post("/predict", data=VALID_INPUT)
    assert response.status_code == 200
    assert b"Predicted class" in response.data
    assert b"Logistic Regression" in response.data or b"Random Forest" in response.data or b"Decision Tree" in response.data or b"KNN" in response.data
    assert b"clinically validated risk percentage" in response.data


def test_missing_input_rejected(client):
    payload = {key: value for key, value in VALID_INPUT.items() if key != "age"}
    response = client.post("/predict", data=payload)
    assert response.status_code == 400
    assert b"This field is required" in response.data


def test_invalid_category_rejected(client):
    response = client.post("/predict", data={**VALID_INPUT, "cp": "99"})
    assert response.status_code == 400
    assert b"Choose a valid option" in response.data


@pytest.mark.parametrize("value", ["banana", "999", "NaN", "54.5"])
def test_invalid_number_rejected(client, value):
    response = client.post("/predict", data={**VALID_INPUT, "age": value})
    assert response.status_code == 400


def test_model_loads_and_exact_feature_contract():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    row, _, errors = validate_input(VALID_INPUT, metadata)
    assert not errors
    assert list(row.columns) == metadata["feature_names"] == list(model.feature_names_in_)
    assert list(row.columns) == [field["name"] for field in FIELDS]
    prediction = model.predict(row)
    probabilities = model.predict_proba(row)
    assert int(prediction[0]) in (0, 1)
    assert probabilities.shape == (1, 2)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-9


def test_submitted_row_is_same_model_prediction(client):
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    row, _, _ = validate_input(VALID_INPUT, metadata)
    expected = int(model.predict(row)[0])
    response = client.post("/predict", data=VALID_INPUT)
    assert f"Predicted class {expected}".encode() in response.data
