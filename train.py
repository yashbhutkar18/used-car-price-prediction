import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

np.random.seed(42)
n_samples = 1200

years = np.random.randint(2010, 2024, size=n_samples)
km_driven = np.random.randint(5000, 150000, size=n_samples)
fuel_types = np.random.choice(["Petrol", "Diesel", "CNG"], size=n_samples, p=[0.55, 0.40, 0.05])
transmissions = np.random.choice(["Manual", "Automatic"], size=n_samples, p=[0.70, 0.30])
engine_cc = np.random.choice([1000, 1200, 1500, 2000, 2500], size=n_samples, p=[0.2, 0.35, 0.25, 0.15, 0.05])
mileage_kmpl = np.random.uniform(12.0, 26.0, size=n_samples)

car_age = 2026 - years
fuel_weight = {"Petrol": 0.0, "Diesel": 0.8, "CNG": -0.4}
trans_weight = {"Manual": 0.0, "Automatic": 1.5}

base_price = (
    (engine_cc * 0.005)
    + (mileage_kmpl * 0.12)
    + np.array([trans_weight[t] for t in transmissions])
    + np.array([fuel_weight[f] for f in fuel_types])
)
depreciation = car_age * 0.45 + (km_driven / 25000.0) * 0.35
noise = np.random.normal(0, 0.4, size=n_samples)
selling_price = np.clip(base_price - depreciation + 3.5 + noise, 1.2, 35.0)

df = pd.DataFrame({
    "Year": years,
    "Kilometers_Driven": km_driven,
    "Fuel_Type": fuel_types,
    "Transmission": transmissions,
    "Engine_CC": engine_cc,
    "Mileage_kmpl": np.round(mileage_kmpl, 1),
    "Selling_Price_Lakh": np.round(selling_price, 2)
})
df.to_csv("data/car_data.csv", index=False)

X = df.drop(columns=["Selling_Price_Lakh"])
y = df["Selling_Price_Lakh"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ["Year", "Kilometers_Driven", "Engine_CC", "Mileage_kmpl"]
categorical_features = ["Fuel_Type", "Transmission"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R2: {r2_score(y_test, y_pred):.3f}")

joblib.dump(pipeline, "model/car_price_model.pkl")
print("Model saved to model/car_price_model.pkl")
