import unittest
from unittest.mock import patch

import app


VALID_INPUT = {
    "age": "55",
    "hypertension": "1",
    "heart_disease": "1",
    "avg_glucose_level": "110",
    "bmi": "28",
    "gender": "Female",
    "ever_married": "No",
    "work_type": "Private",
    "Residence_type": "Urban",
    "smoking_status": "never smoked",
}


class StrokeAppTests(unittest.TestCase):
    def setUp(self):
        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def test_homepage_renders_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Build a snapshot.", response.data)
        self.assertIn(b"avg_glucose_level", response.data)

    def test_valid_submission_shows_prediction_probe_and_recommendations(self):
        response = self.client.post("/", data=VALID_INPUT)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Stroke prediction: Positive", response.data)
        self.assertIn(b"MODEL-ESTIMATED PROBABILITY", response.data)
        self.assertIn(b"XAI PROBE", response.data)
        self.assertIn(b"Suggestions to discuss", response.data)
        self.assertIn(b"cannot tell whether someone is currently having a stroke", response.data)

    def test_prediction_classification_uses_saved_threshold(self):
        with patch("app._positive_probability", return_value=app.THRESHOLD):
            positive = self.client.post("/", data=VALID_INPUT)
        with patch("app._positive_probability", return_value=app.THRESHOLD - 0.001):
            negative = self.client.post("/", data=VALID_INPUT)

        self.assertIn(b"Stroke prediction: Positive", positive.data)
        self.assertIn(b"Stroke prediction: Negative", negative.data)

    def test_invalid_submission_returns_field_error(self):
        invalid = VALID_INPUT | {"age": "-1"}
        response = self.client.post("/", data=invalid)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Enter a value between 0 and 120.", response.data)

    def test_probe_uses_model_fitted_reference_values(self):
        values, errors = app._validate_form(VALID_INPUT)
        self.assertFalse(errors)
        probability = app._positive_probability(values)
        impacts = app._xai_probe(values, probability)
        self.assertEqual(len(impacts), len(app.FEATURES))
        self.assertEqual(impacts, sorted(impacts, key=lambda item: abs(item["impact"]), reverse=True))
        self.assertTrue(all(-1 <= item["impact"] <= 1 for item in impacts))


if __name__ == "__main__":
    unittest.main()
