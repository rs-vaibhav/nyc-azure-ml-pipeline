import azure.functions as func
import json
import logging
import os
import numpy as np
import xgboost as xgb

app = func.FunctionApp()

borough_map = {
    "BRONX": 0, "BROOKLYN": 1, "MANHATTAN": 2,
    "QUEENS": 3, "STATEN ISLAND": 4
}

_model = None

def get_model():
    global _model
    if _model is None:
        _model = xgb.XGBClassifier()
        model_path = os.path.join(os.path.dirname(__file__), "nyc_crime_model.json")
        _model.load_model(model_path)
    return _model

@app.route(route="predict", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Crime prediction request received")
    try:
        req_body = req.get_json()

        borough = req_body.get("borough", "").upper()
        week = req_body.get("week")
        month = req_body.get("month")
        total_complaints = req_body.get("total_complaints", 0)
        unique_complaint_types = req_body.get("unique_complaint_types", 0)
        complaint_crime_ratio = req_body.get("complaint_crime_ratio", 0)

        if borough not in borough_map:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid borough. Must be one of {list(borough_map.keys())}"}),
                status_code=400, mimetype="application/json"
            )

        model = get_model()
        features = np.array([[
            borough_map[borough], week, month,
            total_complaints, unique_complaint_types, complaint_crime_ratio
        ]])

        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        result = {
            "borough": borough,
            "week": week,
            "prediction": "HIGH_CRIME_WEEK" if prediction == 1 else "NORMAL_WEEK",
            "confidence": round(probability, 4),
            "model_version": "xgboost_v1_leakage_free"
        }
        return func.HttpResponse(json.dumps(result), status_code=200, mimetype="application/json")

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return func.HttpResponse(json.dumps({"error": str(e)}), status_code=500, mimetype="application/json")
