import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Dataset load karo
df = pd.read_csv("laptop_data.csv")

# Example expected columns:
# Company, TypeName, Ram, Weight, Touchscreen, Ips, PPI, Cpu_brand,
# HDD, SSD, Gpu_brand, Os, Price

# Features aur target
X = df.drop(columns=["Price"])
y = df["Price"]

# Categorical and numerical columns
categorical_cols = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "Os"]
numerical_cols = ["Ram", "Weight", "Touchscreen", "Ips", "PPI", "HDD", "SSD"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model pipeline
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save model
with open("pipe.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Model saved as pipe.pkl")