from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stroke_model.joblib"
MODEL_ARTIFACT: dict[str, Any] = joblib.load(MODEL_PATH)
PIPELINE = MODEL_ARTIFACT["pipeline"]
THRESHOLD = float(MODEL_ARTIFACT["threshold"])
FEATURES = list(PIPELINE.feature_names_in_)
POSITIVE_CLASS_INDEX = list(PIPELINE.classes_).index(1)

FIELD_CONFIG = {
    "age": {"label": "Age", "type": "number", "min": "0", "max": "120", "step": "1", "suffix": "years"},
    "hypertension": {"label": "Hypertension", "type": "select", "choices": [("0", "No"), ("1", "Yes")]},
    "heart_disease": {"label": "Heart disease", "type": "select", "choices": [("0", "No"), ("1", "Yes")]},
    "avg_glucose_level": {
        "label": "Average glucose level",
        "type": "number",
        "min": "1",
        "max": "2000",
        "step": "any",
        "suffix": "mg/dL",
    },
    "bmi": {"label": "BMI", "type": "number", "min": "1", "max": "100", "step": "any"},
    "gender": {"label": "Gender", "type": "select", "choices": [("Female", "Female"), ("Male", "Male"), ("Other", "Other")]},
    "ever_married": {"label": "Ever married", "type": "select", "choices": [("No", "No"), ("Yes", "Yes")]},
    "work_type": {
        "label": "Work type",
        "type": "select",
        "choices": [
            ("Private", "Private"),
            ("Self-employed", "Self-employed"),
            ("Govt_job", "Government job"),
            ("children", "Child"),
            ("Never_worked", "Never worked"),
        ],
    },
    "Residence_type": {"label": "Residence type", "type": "select", "choices": [("Urban", "Urban"), ("Rural", "Rural")]},
    "smoking_status": {
        "label": "Smoking status",
        "type": "select",
        "choices": [
            ("never smoked", "Never smoked"),
            ("formerly smoked", "Formerly smoked"),
            ("smokes", "Currently smokes"),
            ("Unknown", "Unknown"),
        ],
    },
}

FEATURE_LABELS = {
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "avg_glucose_level": "Average glucose",
    "bmi": "BMI",
    "gender": "Gender",
    "ever_married": "Ever married",
    "work_type": "Work type",
    "Residence_type": "Residence type",
    "smoking_status": "Smoking status",
}

app = Flask(__name__)


def _validate_form(form: Any) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for feature in FEATURES:
        config = FIELD_CONFIG[feature]
        raw_value = form.get(feature, "").strip()
        if not raw_value:
            errors[feature] = "This field is required."
            continue

        if config["type"] == "select":
            allowed = {choice[0] for choice in config["choices"]}
            if raw_value not in allowed:
                errors[feature] = "Choose one of the listed options."
            else:
                values[feature] = int(raw_value) if feature in ("hypertension", "heart_disease") else raw_value
            continue

        try:
            number = float(raw_value)
        except ValueError:
            errors[feature] = "Enter a valid number."
            continue

        minimum = float(config["min"])
        maximum = float(config["max"])
        if not minimum <= number <= maximum:
            errors[feature] = f"Enter a value between {config['min']} and {config['max']}."
        elif config["step"] == "1" and not number.is_integer():
            errors[feature] = "Enter a whole number."
        else:
            values[feature] = int(number) if config["step"] == "1" else number

    return values, errors


def _positive_probability(values: dict[str, Any]) -> float:
    row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}], columns=FEATURES)
    return float(PIPELINE.predict_proba(row)[0, POSITIVE_CLASS_INDEX])


def _xai_probe(values: dict[str, Any], probability: float) -> list[dict[str, Any]]:
    preprocess = PIPELINE.named_steps["prep"]
    numeric_features = MODEL_ARTIFACT["numeric_features"]
    categorical_features = MODEL_ARTIFACT["categorical_features"]
    numeric_reference = dict(
        zip(
            numeric_features,
            preprocess.named_transformers_["num"].named_steps["impute"].statistics_,
        )
    )
    categorical_reference = dict(
        zip(
            categorical_features,
            preprocess.named_transformers_["cat"].named_steps["impute"].statistics_,
        )
    )
    reference = {**numeric_reference, **categorical_reference}

    impacts = []
    for feature in FEATURES:
        comparison = values.copy()
        comparison[feature] = reference[feature]
        comparison_probability = _positive_probability(comparison)
        impact = probability - comparison_probability
        impacts.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "impact": impact,
                "direction": "higher" if impact > 0 else "lower" if impact < 0 else "similar",
                "reference": str(reference[feature]),
            }
        )
    return sorted(impacts, key=lambda item: abs(item["impact"]), reverse=True)


def _recommendations(values: dict[str, Any], impacts: list[dict[str, Any]]) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []
    messages = {
        "hypertension": "Arrange a blood-pressure review with a healthcare professional and follow your existing care plan.",
        "heart_disease": "Discuss this result with your clinician, especially if you have a heart-condition care plan.",
        "avg_glucose_level": "Consider discussing glucose testing and follow-up with a healthcare professional.",
        "bmi": "If weight or nutrition is a concern, ask a clinician about supportive, sustainable options.",
        "smoking_status": "If you smoke, a clinician can help you find a quit-support plan that works for you.",
        "age": "Age cannot be changed; keep up with routine preventive care and discuss personal risk with a clinician.",
    }

    for item in impacts:
        feature = item["feature"]
        if item["impact"] <= 0 or feature not in messages:
            continue
        if feature == "hypertension" and values[feature] != 1:
            continue
        if feature == "heart_disease" and values[feature] != 1:
            continue
        if feature == "smoking_status" and values[feature] != "smokes":
            continue
        suggestions.append(
            {
                "title": f"Follow up on {FEATURE_LABELS[feature].lower()}",
                "body": messages[feature],
            }
        )
        if len(suggestions) == 3:
            break

    if not suggestions:
        suggestions.append(
            {
                "title": "Keep preventive care in view",
                "body": "Review your personal stroke-risk factors with a healthcare professional; this model probe is not a diagnosis.",
            }
        )
    return suggestions


@app.route("/", methods=["GET", "POST"])
def index():
    values: dict[str, Any] = {}
    errors: dict[str, str] = {}
    result = None

    if request.method == "POST":
        values, errors = _validate_form(request.form)
        if not errors:
            probability = _positive_probability(values)
            impacts = _xai_probe(values, probability)
            result = {
                "probability": probability,
                "prediction_positive": probability >= THRESHOLD,
                "threshold": THRESHOLD,
                "impacts": impacts[:5],
                "recommendations": _recommendations(values, impacts),
            }

    return render_template(
        "index.html",
        fields=FIELD_CONFIG,
        values=values,
        errors=errors,
        result=result,
    ), (400 if errors else 200)


if __name__ == "__main__":
    app.run(debug=False)
