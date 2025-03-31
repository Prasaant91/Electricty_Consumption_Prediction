import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.regularizers import l1_l2
import keras_tuner as kt
import seaborn as sns
import numpy as np

data = pd.read_excel("eCO2mix_RTE_Annuel-Definitif_2020 _2021_2022.xlsx")
df = pd.DataFrame(data=data, columns=["Date", "Heures", "Consommation"])
df_cleaned = df.dropna()
df_cleaned = df_cleaned.replace("ND", 0)
df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"])

# Convert "Heures" (time) to string before concatenation
df_cleaned["Heures"] = df_cleaned["Heures"].astype(str)

# Combine Date and Heures into a single datetime column
df_cleaned["Datetime"] = pd.to_datetime(df_cleaned["Date"].dt.strftime("%Y-%m-%d") + " " + df_cleaned["Heures"])

# Print result
pd.set_option('display.max_columns', None)
#df_cleaned = df_cleaned.set_index("Datetime")
scaler = MinMaxScaler(feature_range=(0, 1))
df_cleaned["Consommation"] = scaler.fit_transform(df_cleaned[["Consommation"]])
print(df_cleaned["Consommation"].sum())
df_cleaned["Consommation"].plot()
plt.show()

def create_sequence(data_sequence, sequence_length=48):
    X, y = [], []
    for i in range(len(data_sequence) - sequence_length):
        X.append(data_sequence[i: i + sequence_length])
        y.append(data_sequence[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 48
X, y = create_sequence(df_cleaned["Consommation"].values, sequence_length = sequence_length)

split = int(0.7*len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


# Define function to build an LSTM model with tunable hyperparameters
def build_lstm_model(hp):
    model = Sequential([
        LSTM(
            units=hp.Int("lstm_units", min_value=32, max_value=128, step=32),
            return_sequences=True,
            input_shape=(sequence_length, 1),
            kernel_regularizer=l1_l2(l1=hp.Choice("l1_reg", [0.0001, 0.001, 0.01]),
                                     l2=hp.Choice("l2_reg", [0.0001, 0.001, 0.01]))
        ),
        BatchNormalization(),
        Dropout(hp.Float("dropout_rate", 0.1, 0.5, step=0.1)),
        LSTM(
            units=hp.Int("lstm_units", min_value=32, max_value=128, step=32),
            return_sequences=False,
            kernel_regularizer=l1_l2(l1=hp.Choice("l1_reg", [0.0001, 0.001, 0.01]),
                                     l2=hp.Choice("l2_reg", [0.0001, 0.001, 0.01]))
        ),
        BatchNormalization(),  # Another normalization layer
        Dropout(hp.Float("dropout_rate", 0.1, 0.5, step=0.1)),
        Dense(25, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [0.0001, 0.001, 0.01])
        ),
        loss="mse"
    )
    return model


# Use Keras-Tuner to find the best hyperparameters
tuner = kt.RandomSearch(
    build_lstm_model,
    objective="val_loss",
    max_trials=10,  # Number of different hyperparameter combinations to try
    executions_per_trial=1,  # Number of times to train each model
    directory="lstm_tuning",  # Directory to save tuning results
    project_name="electricity_forecasting_regularization"
)

# Run hyperparameter tuning
tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameter values
print(f"Best LSTM Units: {best_hps.get('lstm_units')}")
print(f"Best Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Best Learning Rate: {best_hps.get('learning_rate')}")

# Train the best model using the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)

y_pred = best_model.predict(X_test)

# Inverse transform the predictions to get original scale values
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))

# Plot actual test values
plt.plot(df_cleaned["Datetime"][-len(y_test_rescaled):], y_test_rescaled, label="Actual Consumption (kWh)", color="blue")

# Plot predicted values
plt.plot(df_cleaned["Datetime"][-len(y_pred_rescaled):], y_pred_rescaled, label="Predicted Consumption (kWh)", color="red", linestyle="dashed")

plt.xlabel("Time")
plt.ylabel("Electricity Consumption (kWh)")
plt.title("LSTM Model: Actual vs. Predicted Electricity Consumption")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Compute RMSE
rmse_value = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f"RMSE: {rmse_value:.3f}")

# Compute R² Score
r2_value = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"R² Score: {r2_value:.3f}")