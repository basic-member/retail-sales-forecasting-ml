import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# ==========================
# Load datasets
# ==========================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
features_df = pd.read_csv("features.csv")
stores_df = pd.read_csv("stores.csv")

# ==========================
# Merge datasets
# ==========================
train = train_df.merge(features_df, on=["Store", "Date"], how="left")
test = test_df.merge(features_df, on=["Store", "Date"], how="left")

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Week"] = train["Date"].dt.isocalendar().week

test["Year"] = test["Date"].dt.year
test["Month"] = test["Date"].dt.month
test["Week"] = test["Date"].dt.isocalendar().week

train = train.drop("Date", axis=1)
test = test.drop("Date", axis=1)

train = train.merge(stores_df, on="Store", how="left")
test = test.merge(stores_df, on="Store", how="left")

# ==========================
# Convert categorical columns
# ==========================
train['IsHoliday_x'] = train['IsHoliday_x'].astype(int)
train['IsHoliday_y'] = train['IsHoliday_y'].astype(int)
test['IsHoliday_x'] = test['IsHoliday_x'].astype(int)
test['IsHoliday_y'] = test['IsHoliday_y'].astype(int)

type_mapping = {'A': 1, 'B': 2, 'C': 3}
train['Type'] = train['Type'].map(type_mapping)
test['Type'] = test['Type'].map(type_mapping)

# ==========================
# Split features / target
# ==========================
X = train.drop("Weekly_Sales", axis=1)
y = train["Weekly_Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# Imputation
# ==========================
num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns

imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

# ==========================
# Models for GridSearch
# ==========================
models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {"fit_intercept": [True, False]}
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(random_state=42),
        "params": {"n_estimators": [50, 100]}
    }
}

best_models = {}

for name, cfg in models.items():
    gs = GridSearchCV(cfg["model"], cfg["params"], cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_models[name] = gs.best_estimator_

# ==========================
# Pick best model by R2
# ==========================
model_scores = {}
for name, model in best_models.items():
    pred = model.predict(X_test)
    model_scores[name] = r2_score(y_test, pred)

best_model_name = max(model_scores, key=model_scores.get)
best_model = best_models[best_model_name]

print(f"Best model selected: {best_model_name}")
final_predictions = best_model.predict(X_test[:6])  # small sample for animation
final_predictions = final_predictions.astype(int)
