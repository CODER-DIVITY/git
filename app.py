# app.py
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('models/lstm_cnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hour = float(request.form['hour'])
    day_of_week = float(request.form['day_of_week'])
    data = np.array([[hour, day_of_week]]).reshape((1, 1, 2))
    prediction = model.predict(data)
    return jsonify({'prediction': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True)
