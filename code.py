import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ======================
# Load sample data
# ======================
df = pd.read_csv("sample_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ======================
# Train / Test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Handle missing values
# ======================
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ======================
# Train model (demo version)
# ======================
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)
model.fit(X_train, y_train)

# ======================
# Predictions
# ======================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Demo Results:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
