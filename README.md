# Heart Disease Prediction System

An SPPU Third Year IT AI and ML mini-project with a complete Jupyter workflow and a Flask prediction interface. The notebook trains and exports a full preprocessing and classification pipeline. Flask loads that saved pipeline; it never retrains a model.

> Educational demonstration only. This application cannot diagnose or rule out heart disease, and its model probability is not a clinically validated medical risk estimate.

## Problem statement and objectives

Use 13 recorded features to classify the [official UCI Heart Disease Cleveland dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) into no disease (`num=0`) versus disease present (`num=1–4`). Inspect missing data and distributions, compare four algorithms fairly, tune on training folds, evaluate once on a held-out test set, and make predictions through a validated web form.

## Dataset

The UCI repository dataset 45 was downloaded with `ucimlrepo` and saved as `dataset/heart_disease.csv`. The local CSV has **303 rows**, **13 input columns**, and the original `num` target. There are **six missing feature cells**: four `ca` and two `thal`. There are **no exact duplicate rows**. The binary target has **164 absent** and **139 present** cases. The dataset is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite Janosi, Steinbrunn, Pfisterer, and Detrano (1989), UCI DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).

The CSV preserves missing values. The training pipeline imputes numeric columns with medians and categorical columns with most frequent values. These are learned on training folds only.

## Machine learning workflow

1. Load and inspect the official local CSV, cleaning `?` and other missing markers.
2. Analyze data types, missing cells, duplicate rows, target balance, histograms, boxplots, categorical distributions, and correlations.
3. Split 80:20 with stratification and random seed 42: 242 training and 61 test rows.
4. Use `ColumnTransformer` and `Pipeline` to impute and scale numeric values and impute and one-hot encode categorical values.
5. Compare Logistic Regression, KNN, Decision Tree, and Random Forest with five-fold stratified cross-validation on training data.
6. Tune each model with a modest `GridSearchCV` parameter grid. Select by training CV ROC-AUC, with F1 as a tie breaker.
7. Evaluate the selected pipeline once on the held-out test set, then export it with Joblib and save JSON metadata.

### Actual results from the executed notebook

Baseline out-of-fold results on the **training set**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.847 | 0.870 | 0.784 | 0.825 | 0.905 |
| Random Forest | 0.818 | 0.807 | 0.793 | 0.800 | 0.893 |
| KNN | 0.822 | 0.821 | 0.784 | 0.802 | 0.868 |
| Decision Tree | 0.698 | 0.667 | 0.685 | 0.676 | 0.697 |

Tuned mean training CV ROC-AUC: Logistic Regression **0.902**, Random Forest **0.901**, KNN **0.881**, Decision Tree **0.836**. Logistic Regression was selected by the stated rule. Its held-out test metrics were **accuracy 0.885**, **precision 0.839**, **recall 0.929**, **F1 0.881**, and **ROC-AUC 0.966**. These scores are specific to one small historical split and do not establish clinical performance.

## Technology stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, Jupyter Notebook, Flask, HTML5, and CSS3. Pytest checks the web application and model contract.

## Folder structure

```text
Heart-Disease-Prediction/
├── dataset/heart_disease.csv
├── notebooks/Complete_ML_Workflow.ipynb
├── notebooks/build_notebook.py
├── models/best_heart_model.pkl
├── models/model_metadata.json
├── templates/{base,index,result,about}.html
├── static/css/style.css
├── tests/test_app.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation

From this project folder, use Python 3.11 or newer. The saved Joblib model was trained with scikit-learn 1.9.1, which is pinned in `requirements.txt` for loading compatibility:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`. The committed CSV and model allow the app to start without downloading data. To download afresh, remove the local CSV and execute the notebook with internet access. If UCI is unavailable, obtain the dataset from the official UCI page and place a CSV with the 13 named feature columns plus `num` at `dataset/heart_disease.csv`.

## Run the notebook

```powershell
python -m jupyter notebook notebooks/Complete_ML_Workflow.ipynb
```

Choose **Run All Cells**. To execute and save all outputs from the command line:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Complete_ML_Workflow.ipynb --ExecutePreprocessor.timeout=600
```

Notebook execution regenerates `models/best_heart_model.pkl` and `models/model_metadata.json`. Use the project folder or its `notebooks` subfolder as the working directory.

## Run Flask

```powershell
python app.py
```

Visit `http://127.0.0.1:5000/`. The routes are `GET /`, `POST /predict`, and `GET /about`. The app validates missing entries, source-supported numeric ranges, and category codes before passing a DataFrame with the exact training feature names to the saved pipeline.

## Example form input and output

| Field | Example |
|---|---:|
| Age | 54 |
| Sex | Male |
| Chest pain | Asymptomatic |
| Resting blood pressure | 130 mm Hg |
| Cholesterol | 246 mg/dl |
| Fasting blood sugar >120 | No |
| Resting ECG | Probable/definite left ventricular hypertrophy |
| Maximum heart rate | 150 bpm |
| Exercise angina | No |
| ST depression | 1.2 |
| ST slope | Flat |
| Major vessels | 1 |
| Thal result | Reversible defect |

The result page reports `Predicted class 0` or `Predicted class 1`, a human-readable label, the selected model name, the model-estimated probability for the predicted class when available, and the submitted values. The class depends on the actual saved pipeline.

## Test

```powershell
python -m pytest -q
```

Tests cover the home and about pages, valid predictions, missing and invalid fields, model loading, and the exact 13-feature notebook-to-Flask contract.

## Screenshots

For a practical presentation, run Flask and capture the home form, result page, and about page in a browser. The executed notebook also contains the EDA charts, confusion matrices, ROC curves, and feature score plot.

## Limitations and medical disclaimer

The Cleveland data is small, old, and from a specific source population. Its labels and measurements do not represent every patient or modern clinical workflow. The model is not externally validated or probability calibrated. Dataset-supported form ranges are demonstration bounds, not medically normal ranges. No prediction or probability from this project should guide diagnosis, treatment, or emergency decisions. Consult a qualified clinician for medical advice.
