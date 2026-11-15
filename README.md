# Stroke Signal

A Flask demo for exploring the supplied `stroke_model.joblib` pipeline. It displays the model-estimated positive-class probability, compares it with the decision threshold stored in the artifact, and runs a simple one-feature-at-a-time probe against the pipeline's fitted median/mode imputation values.

The probe is a lightweight model explanation, not SHAP, a causal analysis, or medical advice. Its recommendations are general prompts for discussion with a healthcare professional. This app is not a diagnosis or screening test. For possible stroke symptoms, contact emergency services immediately.

## Run locally

The model was saved with scikit-learn 1.7.2. Use that version to avoid pickle compatibility problems:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. Keep `stroke_model.joblib` in the project root beside `app.py`.

## Deploy on Render

1. Push this project to a GitHub repository. Include `stroke_model.joblib` in the repository root beside `app.py`.
2. In Render, choose **New → Blueprint**, connect the repository, and apply the `render.yaml` configuration. It explicitly selects the **Free** plan. Alternatively, create a **Web Service**, choose **Free** as its compute plan, and use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
3. Wait for the build and deploy to finish, then open the `onrender.com` URL Render provides.

The `.python-version` file selects Python 3.12, and `requirements.txt` pins scikit-learn to the version used to save the model. Free web services spin down after 15 minutes without traffic, so the first request after idle may take about a minute while the service starts. The deployed model file is part of the app; do not rely on the service filesystem to persist user data. Free instances are intended for demos and hobby projects, not production or medical decision-making.

## Tests

```powershell
python -m unittest discover -s tests
```
