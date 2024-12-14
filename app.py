import joblib
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = load_model('EV_Regression_Model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])

        # Prepare input data as a 2D array (even for one sample)
        input_data = np.array([[feature1, feature2, feature3, feature4]])

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the trained model
        prediction = model.predict(input_data_scaled)

        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
