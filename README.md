**Time Series Forecasting Projects
**

This repository contains three complete machine learning projects designed to forecast electricity consumption from time series data collected at 30-minute intervals. The goal was to explore, compare, and understand different modeling approaches for time series forecasting:

LSTM (Deep Learning using recurrent neural networks)

TimeGPT (Zero-shot, large foundation model for time series)

XGBoost (Tree-based gradient boosting with exogenous features)

Each model has been implemented with a full pipeline from data preprocessing, feature engineering, model training, evaluation, and visualization.
Diverse Approaches to Time Series Forecasting

**1. LSTM (Long Short-Term Memory Neural Network)
**
The LSTM model was designed to learn from the sequential behavior of electricity consumption data. Since the data was recorded every 30 minutes, we structured it into daily sequences of 48 time steps. To prepare this data, we first normalized the consumption values using MinMaxScaler to ensure the LSTM network could learn efficiently. We then reshaped the dataset into sequences (X) of past 48 steps and corresponding labels (y) for the next step. This way, each sample gave the model a full day's worth of consumption to predict the next half-hour.

Once the data was structured, we split it into training (80%) and testing (20%) sets. We built a Keras-based LSTM model consisting of one LSTM layer with 50 units, followed by a dense output layer. We trained it over multiple epochs, monitoring loss using Mean Squared Error (MSE) and optimizing with Adam optimizer. After training, predictions were made on the test data. We plotted the actual vs predicted values to visually evaluate performance and computed RMSE and R² scores to quantify model accuracy.

**2. TimeGPT (Nixtla’s Zero-Shot Foundation Model)
**
The TimeGPT model is a transformer-based foundation model trained by Nixtla for time series forecasting. It works in a zero-shot fashion, meaning no model training is required on the local dataset. To use TimeGPT, we first ensured that our dataset followed the required format: ds for timestamps, y for the target variable (consumption), and unique_id for series identifier. We also made sure the timestamp column was continuous and gap-free with a frequency of 30 minutes.

We then split our dataset into 80% for training context and 20% for forecasting. Using the NixtlaClient, we passed in the training portion along with parameters for forecast horizon (equal to the length of the test set), frequency (30min), and confidence levels (80% and 90%). TimeGPT returned a forecasted dataframe which included predicted values and upper/lower bounds of prediction intervals. We plotted the results with matplotlib, comparing them against actual values and also computed RMSE and MAE to evaluate forecast quality.

The major advantage here was not needing to build or train a model—TimeGPT used transfer learning from massive time series data and worked out-of-the-box.

**3. XGBoost Regressor (Gradient Boosting with Lag & Temporal Features)
**
The XGBoost model was developed using a traditional machine learning approach with heavy feature engineering. We started by extracting time-related features from the timestamp column, such as hour, day, day of the week, month, and whether it was a weekend. These categorical features helped the model recognize patterns such as peak hours or weekend dips in consumption.

We then added lag-based features to give the model memory of past behavior. These included lags of various lengths: lag_1, lag_3, lag_6, lag_12, lag_24 (one day), and lag_336 (one week). This allowed XGBoost to learn short-term and long-term dependencies. Before training, we dropped any rows with missing values from lagging and ensured only numeric features were passed to the model.

The dataset was then split into 80% training and 20% testing sets. We trained an XGBRegressor model initially with default parameters. Then, to improve performance, we tuned the model using two methods: Grid Search (trying out all combinations) and Bayesian Optimization with Optuna, which intelligently explored the parameter space. The best model was selected based on lowest RMSE from validation data.

We evaluated the model by computing RMSE and R², and visualized the predicted vs actual values. We also used XGBoost's feature importance method to see which lag or time features were most influential. Furthermore, we created scatter plots of lag features vs target to analyze their linear relationship and added category-specific coloring and marker shapes for clarity.

**Visualization Techniques Used
**
We used matplotlib and seaborn extensively to visualize the model outputs and relationships in the data. Time series plots showed how well predicted values followed actual trends. Heatmaps displayed correlation between lag features and the target variable. Scatter plots revealed relationships between lagged features (e.g., 1 day ago, 1 week ago) and the current target. We enhanced these plots using dynamic marker shapes and color palettes to reflect additional variables like hour of day or weekday/weekend distinction.

**Future Enhancements
**
Introduce rolling mean and standard deviation as additional features

Apply seasonal decomposition (STL) or Fourier transformations

Combine multiple models in an ensemble approach

Serve model as a REST API or integrate with Streamlit dashboards for real-time useFuture Enhancements

Introduce rolling mean and standard deviation as additional features

Apply seasonal decomposition (STL) or Fourier transformations

Combine multiple models in an ensemble approach

Serve model as a REST API or integrate with Streamlit dashboards for real-time use



