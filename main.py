import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Module 1 — Data Collection and Loading
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


# Module 2 — Data Cleaning
def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df


# Module 3 — Feature Selection
def select_features(df):
    df = df[
        [
            "City",
            "Locality_Tier",
            "Super_Area_sqft",
            "Carpet_Area_sqft",
            "Furnishing",
            "Crime_Rate_Index",
            "Market_Price_INR"
        ]
    ]
    return df


# Module 4 — Encoding
def encode_data(df):
    df = pd.get_dummies(
        df,
        columns=["City", "Locality_Tier", "Furnishing"],
        drop_first=True,
        dtype=int
    )
    return df


# Module 5 — Data Preparation
def split_features_target(df):
    X = df.drop("Market_Price_INR", axis=1)
    y = df["Market_Price_INR"]
    return X, y


# Module 6 — Split Dataset
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


# Module 7 — Model Training
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# Module 8 — Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance")
    print("Model Used: Random Forest Regression")
    print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
    print(f"R² Score: {round(r2, 4)}")


# Module 9 — Feature Importance Analysis
def feature_importance(model, X):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== Feature Importance ==========")
    print(importance)


# Module 10 — Save Model
def save_model(model):
    joblib.dump(model, "house_price_model.pkl")
    print("\nModel Saved Successfully")
    print("File Name : house_price_model.pkl")


# Main / Driver Code
def main():
    df = load_data("house_price_dataset.csv")
    df = select_features(df)
    df = clean_data(df)
    df = encode_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    feature_importance(model, X)
    save_model(model)


if __name__ == "__main__":
    main()