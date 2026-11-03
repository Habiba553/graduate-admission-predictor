# Graduate Admission Predictor — Streamlit deployment

This folder contains a starter Streamlit interface for the admission-prediction notebook.

## Important before deployment

The app is configured for Variant A1 from the notebook: `GPA Score`, `is_international`, and `University rating`. This is the reduced-feature model intended to work with the broader 2025–2026 test set. It does **not** use GRE because GRE information is missing or imputed for many recent records.

The app will not run predictions until you export `admission_model.joblib` from your notebook and put it beside `app.py`.

## 1. Export the model from your notebook

After the data-cleaning cells have created `temporal_df`, run this in a new notebook cell. It refits the selected A1 Random Forest on all eligible historical training rows and saves the fitted model.

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

features_a1 = ["GPA Score", "is_international", "University rating"]
train_a1 = temporal_df.dropna(subset=features_a1 + ["Admitted"]).copy()

X_final = train_a1[features_a1]
y_final = train_a1["Admitted"].astype(int)

deployment_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)
deployment_model.fit(X_final, y_final)

joblib.dump(
    {"model": deployment_model, "features": features_a1},
    "admission_model.joblib"
)
print("Saved admission_model.joblib")
```

**Check before running:** Confirm that `temporal_df["Admitted"]` uses `1 = Accepted` and `0 = Rejected`, and that `University rating` uses the same scale/mapping as in training. This export snippet assumes those conditions are true.

Download `admission_model.joblib` from your notebook environment and place it in this folder. Do not upload the notebook's full raw dataset to a public repository unless you have permission to redistribute it.

## 2. Test locally

Install Python, then open a terminal in this folder:

```bash
python -m venv .venv
```

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the app locally.

## 3. Deploy on Streamlit Community Cloud

1. Create a GitHub repository for this app.
2. Upload `app.py`, `requirements.txt`, and `admission_model.joblib`.
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
5. Deploy and test the public URL.

Keep the model and app compatible with the same versions of scikit-learn. Only upload data/model files that you are allowed to share.

## Suggested repository structure

```text
admission-predictor/
├── app.py
├── requirements.txt
├── admission_model.joblib
└── README.md
```

## Limitations to disclose

- This is an experimental estimate, not an official admission decision.
- GradCafe submissions can be self-selected and may not represent all applicants.
- Admissions policies change over time; temporal validation is important.
- A model probability is not necessarily a calibrated real-world probability.
