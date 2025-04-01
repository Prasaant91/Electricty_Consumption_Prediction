import pandas as pd
from nixtlats import NixtlaClient
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from utilsforecast.losses import mse

api_key = " "
client = NixtlaClient(api_key=api_key)
data = pd.read_excel("eCO2mix_RTE_Annuel-Definitif_2020 _2021_2022.xlsx")
df = pd.DataFrame(data=data, columns=["Date", "Heures", "Consommation"])
df_cleaned = df.dropna()
df_cleaned = df_cleaned.replace("ND", 0)
df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])
df_cleaned = df_cleaned.drop_duplicates()

# Convert "Heures" (time) to string before concatenation
df_cleaned["Heures"] = df_cleaned["Heures"].astype(str)

# Combine Date and Heures into a single datetime column
df_cleaned["Datetime"] = pd.to_datetime(df_cleaned["Date"].dt.strftime("%Y-%m-%d") + " " + df_cleaned["Heures"])
df_cleaned = df_cleaned.drop("Heures", axis='columns')
df_cleaned = df_cleaned.drop("Date", axis='columns')
df_cleaned.rename(columns={'Datetime': 'ds'}, inplace=True)
# Print result
pd.set_option('display.max_columns', None)
#df_cleaned = df_cleaned.set_index("Datetime")

split_idx = int(0.8 * len(df_cleaned))
train_df = df_cleaned.iloc[:split_idx]
test_df = df_cleaned.iloc[split_idx:]
print(train_df.columns)

forecast_df = client.forecast(
    df=train_df,
    h=480,
    time_col = "ds",
    target_col = "Consommation",
    freq="30min",
    finetune_steps=1,
    finetune_loss='mse',
    model='timegpt-1-long-horizon',
    level=[80, 90]
)
print(forecast_df.columns)

forecast_df["ds"] = test_df["ds"].values[:len(forecast_df)]

# Plot actual consumption
plt.plot(test_df["ds"], test_df["Consommation"], label="Actual", color="blue")

# Plot forecasted consumption
plt.plot(forecast_df["ds"], forecast_df["TimeGPT"], label="Forecast", color="red", linestyle="--")

# Plot confidence intervals (optional)
plt.fill_between(forecast_df["ds"], forecast_df["TimeGPT-lo-90"], forecast_df["TimeGPT-hi-90"], color="red", alpha=0.1, label="90% CI")
plt.fill_between(forecast_df["ds"], forecast_df["TimeGPT-lo-80"], forecast_df["TimeGPT-hi-80"], color="red", alpha=0.2, label="80% CI")

# Formatting
plt.title("TimeGPT Forecast vs Actual (Electricity Consumption)")
plt.xlabel("Time")
plt.ylabel("Consumption")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
