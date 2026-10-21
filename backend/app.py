import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask("superkart")

# Load the trained SuperKart model (the .joblib file sits next to this script in the container)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model_v1_0.joblib")
model = joblib.load(MODEL_PATH)

# Features expected by the model
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SuperKart Sales Prediction API"})


@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Please provide a valid JSON object."}), 400

        missing = [col for col in FEATURES if col not in data]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        # Convert the extracted data into a DataFrame
        input_data = pd.DataFrame([[data[col] for col in FEATURES]], columns=FEATURES)

        # Make a sales prediction using the trained model
        prediction = model.predict(input_data)

        return jsonify({
            "predicted_Product_Store_Sales_Total": float(prediction[0])
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Please upload a CSV file using the 'file' field."}), 400

        file = request.files["file"]

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a CSV file."}), 400

        input_df = pd.read_csv(file)

        missing = [col for col in FEATURES if col not in input_df.columns]
        if missing:
            return jsonify({
                "error": "CSV is missing required features",
                "missing_features": missing
            }), 400

        # Select features in the expected order
        input_df = input_df[FEATURES]

        predictions = model.predict(input_df)

        result = input_df.copy()
        result["Predicted_Product_Store_Sales_Total"] = predictions

        return jsonify({"predictions": result.to_dict(orient="records")})

    except Exception as e:
        app.logger.exception("Batch prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
