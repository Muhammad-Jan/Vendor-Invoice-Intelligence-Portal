import joblib
import pandas as pd

MODEL_PATH = "invoice_flagging/models/predict_flag_invoice.pkl"
SCALER_PATH = "invoice_flagging/models/scaler.pkl"


def load_model(model_path=MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def load_scaler(scaler_path=SCALER_PATH):
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler


def predict_invoice_flag(input_data):

    model = load_model()
    scaler = load_scaler()

    input_df = pd.DataFrame(input_data)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    input_df["Predicted_Flag"] = prediction

    return input_df


if __name__ == "__main__":

    sample_data = {
        "invoice_quantity": [50],
        "invoice_dollars": [352.95],
        "Freight": [1.73],
        "total_item_quantity": [162],
        "total_item_dollars": [2476]
    }

    prediction = predict_invoice_flag(sample_data)

    print(prediction)