# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load data
data = pd.read_csv('data/raw_data.csv')

# Data cleaning
data = data.dropna()  # Dropping missing values

# Feature extraction
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek

# Normalizing data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[['traffic_density', 'hour', 'day_of_week']])

# Save processed data
processed_data = pd.DataFrame(scaled_features, columns=['traffic_density', 'hour', 'day_of_week'])
processed_data.to_csv('data/processed_data.csv', index=False)
