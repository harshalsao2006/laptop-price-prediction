import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Dataset load
df = pd.read_csv("laptop_data.csv")

# Example simple features
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open("pipe.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")