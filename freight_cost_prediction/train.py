import joblib
from pathlib import Path

from data_preprocessing import (
    load_vendor_invoice_data,
    prepare_features,
    split_data
)

from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

db_path = BASE_DIR / "Data" / "inventory.db"
model_dir = BASE_DIR / "models"


# ------------------------------------------------------------
# Main Training Pipeline
# ------------------------------------------------------------

def main():

    # Create models folder if it doesn't exist
    model_dir.mkdir(exist_ok=True)

    # Load dataset
    df = load_vendor_invoice_data(db_path)

    # Prepare features & target
    X, y = prepare_features(df)

    print("\nTraining Features:", list(X.columns))

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate Models
    results = []

    results.append(
        evaluate_model(
            lr_model,
            X_test,
            y_test,
            "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            dt_model,
            X_test,
            y_test,
            "Decision Tree Regression"
        )
    )

    results.append(
        evaluate_model(
            rf_model,
            X_test,
            y_test,
            "Random Forest Regression"
        )
    )

    # Select Best Model (Lowest MAE)
    best_model_info = min(results, key=lambda x: x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    # Save Best Model
    model_path = model_dir / "predict_freight_model.pkl"

    joblib.dump(best_model, model_path)

    print("\n----------------------------------------")
    print(f"Training Features : {list(X.columns)}")
    print(f"Best Model        : {best_model_name}")
    print(f"Model Saved At    : {model_path}")
    print("----------------------------------------")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    main()