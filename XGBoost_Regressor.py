import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

palette = sns.color_palette("husl", 24)

data = pd.read_excel("eCO2mix_RTE_Annuel-Definitif_2020 _2021_2022.xlsx")
df = pd.DataFrame(data=data, columns=["Date", "Heures", "Consommation"])
df_cleaned = df.dropna()
df_cleaned = df_cleaned.replace("ND", 0)
df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])

# Convert "Heures" (time) to string before concatenation
df_cleaned["Heures"] = df_cleaned["Heures"].astype(str)

# Combine Date and Heures into a single datetime column
df_cleaned["Datetime"] = pd.to_datetime(df_cleaned["Date"].dt.strftime("%Y-%m-%d") + " " + df_cleaned["Heures"])

df_cleaned["hour"] = df_cleaned["Datetime"].dt.hour
df_cleaned["day"] = df_cleaned["Datetime"].dt.day
df_cleaned["dayofweek"] = df_cleaned["Datetime"].dt.dayofweek
df_cleaned["month"] = df_cleaned["Datetime"].dt.month
df_cleaned["minute"] = df_cleaned["Datetime"].dt.minute
df_cleaned["is_weekend"] = df_cleaned["Datetime"].isin([5, 6]).astype(int)

lags_to_check = {
    #"1_hour": 2,
    #"3_hours": 6,
    #"6_hours": 12,
    #"12_hours": 24,
    "1_day": 48
    #"2_days": 96,
    #"1_week": 336,
    #"1_month": 336*4
}

for name, lag in lags_to_check.items():
    df[f"lag_{name}"] = df_cleaned["Consommation"].shift(lag)

feature_cols = [col for col in df_cleaned.columns if col not in ["Datetime", "Consommation"]]
print(feature_cols)

X = df_cleaned[feature_cols]
Y = df_cleaned["Consommation"]

X = X.drop(columns=["Date", "Heures"])

split_idx = int(0.8 * len(df_cleaned))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
timestamp_test = df_cleaned["Datetime"].iloc[split_idx:]

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,                          # 3-fold cross-validation
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print(f" RMSE: {rmse:.2f}")
print(f" R² Score: {r2:.4f}")

plt.figure(figsize=(14, 6))
plt.plot(timestamp_test, Y_test, label="Actual", color="blue")
plt.plot(timestamp_test, Y_pred, label="Tuned XGBoost", color="red", linestyle="--")
plt.title("Tuned XGBoost Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

importance = best_model.feature_importances_
features = X_train.columns
sorted_idx = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], importance[sorted_idx])
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()