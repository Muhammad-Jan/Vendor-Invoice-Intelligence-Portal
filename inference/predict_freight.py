import joblib
import pandas as pd

MODEL_PATH = "freight_cost_prediction/models/predict_freight_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict or pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    model = load_model()

    # Convert input to DataFrame if needed
    if isinstance(input_data, dict):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()

    # Ensure feature order matches training
    input_df = input_df[["Quantity", "Dollars"]]

    # Predict
    input_df["Predicted_Freight"] = model.predict(input_df).round(2)

    return input_df


if __name__ == "__main__":

    # Example inference run
    sample_data = {
        "Quantity": [1200, 500],
        "Dollars": [18500, 5000]
    }

    prediction = predict_freight_cost(sample_data)

    print(prediction)