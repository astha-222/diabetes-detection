from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(request.form[field]) for field in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]
    
    # Convert and reshape for prediction
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)

    # Interpret prediction
    result = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
