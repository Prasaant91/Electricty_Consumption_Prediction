import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import optuna
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
print(df_cleaned["is_weekend"])
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

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'random_state': 42
    }

    # Train model with suggested parameters
    model = XGBRegressor(**params)
    model.fit(X_train, Y_train)

    # Predict and evaluate RMSE
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

    return rmse

study = optuna.create_study(direction='minimize')  # minimize RMSE
study.optimize(objective, n_trials=50, timeout=600)  # 50 trials or 10 minutes

print("Best Parameters:", study.best_params)
print("Best RMSE:", study.best_value)

# Train best model
best_model = XGBRegressor(**study.best_params)
best_model.fit(X_train, Y_train)

Y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print(f" Final RMSE: {rmse:.2f}")
print(f" Final R²: {r2:.4f}")

optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

plt.figure(figsize=(14, 6))
plt.plot(timestamp_test, Y_test, label="Actual", color="blue", linewidth=2)
plt.plot(timestamp_test, Y_pred, label="Predicted", color="red", linestyle="--", linewidth=2)

plt.title("XGBoost Forecast vs Actual")
plt.xlabel("Timestamp")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()