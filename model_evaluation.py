# model_evaluation.py
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load the trained model
model = load_model('models/lstm_cnn_model.h5')

# Predict using the model
y_pred = model.predict(X_test)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Traffic Prediction')
plt.xlabel('Time')
plt.ylabel('Traffic Density')
plt.legend()
plt.show()
