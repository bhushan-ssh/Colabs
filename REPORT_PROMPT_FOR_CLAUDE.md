# Copy this entire prompt into Claude

Create a polished, original **PDF project report** for my SPPU (Savitribai Phule Pune University) Third Year Information Technology AI and ML laboratory mini-project. The title is **Heart Disease Prediction System Using Machine Learning and Flask**. Deliver the finished PDF as a downloadable file, not just a plan or Markdown. If your environment supports an editable source document, provide that too. Aim for **12 pages** and keep the final PDF between **10 and 15 pages inclusive**, counting the cover, contents, and references.

I can upload the actual project files. Use `Complete_ML_Workflow.ipynb`, `model_metadata.json`, `README.md`, `app.py`, and any screenshots I attach to verify the details below. The factual summary here is sufficient to write the report if the files are not accessible. Do not invent results, medical claims, screenshots, citations, student details, or implementation features. If a required personal detail is missing, keep the clearly marked placeholder.

## Cover details to keep as placeholders

- Student name: **[STUDENT NAME]**
- PRN: **[PRN]**
- Roll number: **[ROLL NUMBER]**
- Division: **[DIVISION]**
- College name: **[COLLEGE NAME]**
- Department: **Information Technology**
- University: **Savitribai Phule Pune University (SPPU)**
- Guide/faculty: **[GUIDE NAME]**
- Academic year: **[ACADEMIC YEAR]**
- Submission date: **[SUBMISSION DATE]**

Do not create a fake college logo, signature, certificate, acknowledgement, or approval page. Use a simple professional cover that can later be edited with my details.

## Report structure and page budget

Use the following as a target layout; adjust page breaks for readability while staying within 10–15 pages.

1. **Cover page**: title, project type, placeholders above.
2. **Abstract and table of contents**: 150–200 word abstract, keywords, compact contents with correct page numbers.
3. **Introduction, problem statement, objectives**: educational purpose, binary classification task, three to five concrete objectives.
4. **Dataset and feature dictionary**: provenance, target mapping, record count, missingness, concise table covering all 13 features and valid categorical codes.
5. **Methodology and preprocessing**: cleaning, 80:20 split, numerical/categorical pipeline, leakage prevention, cross-validation and model selection rule. Include one clear workflow diagram.
6. **Exploratory data analysis**: target distribution and two or three informative plots sourced from the executed notebook if available. Explain what the plots show without causal claims. If plots cannot be extracted, use a truthful target count chart from the provided numbers and describe other EDA without pretending to show them.
7. **Algorithms and model comparison**: short understandable descriptions of Logistic Regression, KNN, Decision Tree, and Random Forest, with actual baseline and tuned training results.
8. **System architecture and implementation**: notebook export to Joblib pipeline to Flask form; routes; validation; exact feature contract. Include a small architecture diagram.
9. **Results and discussion**: final model, held-out test metrics, confusion matrix with interpretation, distinction between training CV and test results.
10. **Web application and testing**: form sections, result view, validation, 11 automated tests and live route checks. Include real screenshots only if I provide them. Otherwise use a simple labeled interface schematic or omit screenshots.
11. **Limitations, conclusion, future scope**: careful discussion of small historical dataset, lack of external/clinical validation and probability calibration; practical learning outcomes; realistic future work.
12. **References and optional concise appendix**: correctly formatted sources and, only if space allows, execution commands or a compact feature schema.

Write in clear academic English suitable for a Third Year IT student to explain in a practical examination. Use numbered headings, page numbers, figure/table numbers and captions, consistent fonts, generous but economical spacing, and no oversized blank areas. The report should contain explanation and analysis, not pages of pasted code. Use short code or pseudocode excerpts only where they explain the pipeline or Flask flow. Make all tables legible on A4 pages. Check the generated PDF visually and correct overflow, clipped tables, bad page breaks, missing symbols, or wrong page count before delivering it.

## Verified project facts

### Purpose and source

The application predicts the binary target from the **official UCI Heart Disease Cleveland dataset**, repository dataset ID 45: https://archive.ics.uci.edu/dataset/45/heart+disease . The original target `num` has values 0–4. The project maps `0` to no heart disease recorded and `1, 2, 3, 4` to heart disease present. This is a machine learning demonstration, not a clinical decision tool.

The data was fetched through the `ucimlrepo` Python package and saved locally as `dataset/heart_disease.csv`. There are **303 source and cleaned rows**, **13 input features**, **164 target-0 rows**, and **139 target-1 rows**. There are **six missing feature cells**: four in `ca` and two in `thal`. There are **zero exact duplicate rows**. Missing values remain in the CSV and are handled inside the fitted preprocessing pipeline.

The observed numerical source ranges, used by the web form for demonstration validation, are: age 29–77 years, resting blood pressure (`trestbps`) 94–200 mm Hg, cholesterol (`chol`) 126–564 mg/dl, maximum heart rate (`thalach`) 71–202 bpm, and exercise-related ST depression (`oldpeak`) 0.0–6.2. These are observed dataset ranges, **not normal or medically safe ranges**.

### Feature dictionary

All 13 input names must appear exactly as below. Describe codes using the UCI Cleveland definitions; do not call `thal` a thalassemia diagnosis.

| Feature | Type | Meaning / valid codes |
|---|---|---|
| `age` | Numeric | Age in years |
| `sex` | Categorical | Recorded sex: 0 female, 1 male |
| `cp` | Categorical | Chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `trestbps` | Numeric | Resting blood pressure on hospital admission, mm Hg |
| `chol` | Numeric | Serum cholesterol, mg/dl |
| `fbs` | Categorical | Fasting blood sugar >120 mg/dl: 0 false, 1 true |
| `restecg` | Categorical | Resting ECG: 0 normal, 1 ST-T wave abnormality, 2 probable/definite left ventricular hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved |
| `exang` | Categorical | Exercise-induced angina: 0 no, 1 yes |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping |
| `ca` | Categorical | Number of major vessels colored by fluoroscopy: 0, 1, 2, 3 |
| `thal` | Categorical | Thal result: 3 normal, 6 fixed defect, 7 reversible defect |

### Exact ML procedure

- The executed notebook is `notebooks/Complete_ML_Workflow.ipynb`. It contains 46 numbered workflow sections, 88 cells in total, and 42 executed code cells with no execution errors.
- Python libraries: Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, ucimlrepo, Jupyter; Flask serves the UI. Pytest runs the automated app tests.
- The cleaned data is split **80:20, stratified**, using `random_state=42`: **242 training rows**, **61 test rows**. The test target support is **33 absent** and **28 present**.
- Numerical features: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`. Pipeline applies `SimpleImputer(strategy="median")`, then `StandardScaler()`.
- Categorical features: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`. Pipeline applies `SimpleImputer(strategy="most_frequent")`, then `OneHotEncoder(handle_unknown="ignore")`.
- A `ColumnTransformer` combines both branches and is inside a scikit-learn `Pipeline` with each classifier. Every cross-validation fold fits preprocessing only on its own training portion. The held-out test set is not used for model selection or hyperparameter tuning.
- Four classifiers were trained and compared: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest.
- Baseline comparison used out-of-fold predictions on training data. It reports accuracy, precision, recall, F1 and ROC-AUC, plus confusion matrices and ROC curves for all four models.
- Hyperparameter tuning used `GridSearchCV` with **five-fold shuffled stratified cross-validation**, `random_state=42`, and modest parameter grids. The primary selection metric was mean training CV ROC-AUC, with F1 as a tie breaker. The selected Logistic Regression parameter was `C=1.0`.
- The selected *complete fitted preprocessing + classifier pipeline* was exported to `models/best_heart_model.pkl`; feature names, types, categories, ranges, target mapping, model name, and metrics were saved to `models/model_metadata.json`.

### Actual model comparison results

**Baseline out-of-fold training results** (do not label them as test results):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.847 | 0.870 | 0.784 | 0.825 | 0.905 |
| Random Forest | 0.818 | 0.807 | 0.793 | 0.800 | 0.893 |
| KNN | 0.822 | 0.821 | 0.784 | 0.802 | 0.868 |
| Decision Tree | 0.698 | 0.667 | 0.685 | 0.676 | 0.697 |

**Tuned five-fold mean training CV results** (different evaluation procedure from the baseline table):

| Model | CV ROC-AUC | CV F1 | CV Recall |
|---|---:|---:|---:|
| Logistic Regression | 0.902 | 0.825 | 0.783 |
| Random Forest | 0.901 | 0.781 | 0.764 |
| KNN | 0.881 | 0.799 | 0.783 |
| Decision Tree | 0.836 | 0.743 | 0.729 |

**Selected final model: Logistic Regression.** It had the highest tuned mean training CV ROC-AUC, **0.902493**. Random Forest was close at **0.901268**, so do not overstate superiority; the small difference may be unstable across splits. The model was selected before viewing the held-out test metrics.

**Held-out test evaluation for selected Logistic Regression**, 61 rows:

| Metric | Actual value |
|---|---:|
| Accuracy | 0.8852459 |
| Precision (disease-present class) | 0.8387097 |
| Recall (disease-present class) | 0.9285714 |
| F1 (disease-present class) | 0.8813559 |
| ROC-AUC | 0.9664502 |

Confusion matrix, rows = actual `[absent, present]`, columns = predicted `[absent, present]`:

```text
                 Predicted absent   Predicted present
Actual absent           28                  5
Actual present           2                 26
```

Thus TN=28, FP=5, FN=2, TP=26. Discuss the two missed disease-present cases without claiming clinical safety. The 0.966 test ROC-AUC comes from this small holdout and should not be portrayed as external validation.

### Flask application

- `app.py` loads the exported Joblib pipeline and JSON metadata at startup. It does **not** train a separate model.
- Routes: `GET /` renders the 13-field form; `POST /predict` validates input, constructs a one-row DataFrame in the **exact notebook feature order**, calls the saved pipeline, and renders the result; `GET /about` explains the project and dataset.
- Form sections: personal information, symptoms/history, measurements, exercise test, clinical tests. Numeric entries are checked for valid finite numbers and observed source ranges. Categorical entries must match valid UCI codes. Missing fields are rejected with field-level messages.
- Result page shows predicted class, readable label, selected model name, submitted values, and model probability for the predicted class if `predict_proba` is available. This is described as model-estimated probability, **not a clinically validated risk percentage**. Both pages contain an educational medical disclaimer.
- Frontend uses responsive HTML5 and custom CSS; no complex JavaScript or frontend framework.
- The automated test suite has **11 passing tests**. It checks the home and about pages, valid prediction, missing input, invalid categorical and numeric input, model loading, exact feature contract, and consistency between direct pipeline prediction and the web result. The running Flask server was also checked over HTTP: home, about, and valid prediction each returned HTTP 200.

### Project files

```text
Heart-Disease-Prediction/
├── dataset/heart_disease.csv
├── notebooks/Complete_ML_Workflow.ipynb
├── notebooks/build_notebook.py
├── models/best_heart_model.pkl
├── models/model_metadata.json
├── templates/base.html
├── templates/index.html
├── templates/result.html
├── templates/about.html
├── static/css/style.css
├── tests/test_app.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The notebook should be run before Flask when retraining. Typical commands from the project directory are:

```powershell
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/Complete_ML_Workflow.ipynb --ExecutePreprocessor.timeout=600
python app.py
python -m pytest -q
```

The saved model uses scikit-learn 1.9.1; the requirements file pins that version for Joblib compatibility.

## Writing and citation rules

- Clearly separate **baseline training out-of-fold scores**, **tuned mean training CV scores**, and **one held-out test evaluation**. Never present the four-model training comparison as four test-set results.
- Explain what accuracy, precision, recall, F1, ROC-AUC and each confusion-matrix cell mean in this binary task, briefly and correctly.
- Include one diagram of the ML pipeline and one diagram of the Flask prediction flow. Create them from the verified architecture above; these are conceptual diagrams, not screenshots.
- Cite the UCI source for data and code definitions. The source page gives: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X . Cite official scikit-learn and Flask documentation only where they genuinely support technical descriptions. Do not fabricate articles or DOI numbers. Put citations in the text and a matching reference list at the end.
- Avoid claiming the model detects heart attacks, predicts future disease, proves clinical usefulness, or gives a medical risk score. It classifies the dataset's recorded disease label from the 13 inputs.
- Include a clear limitation: small historical dataset, one source population, no external validation, no clinical validation, and no probability calibration. The application is for educational use only and should not guide diagnosis or treatment.
- Before final delivery, verify the PDF is 10–15 pages, all figures/tables are visible, page numbers and table of contents are correct, and placeholders remain obvious for my personal details.
