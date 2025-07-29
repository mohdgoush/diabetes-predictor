from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    form_data = {
        "pregnancies": 0,
        "glucose": 100,
        "bp": 70,
        "skin": 20,
        "insulin": 80,
        "bmi": 25.0,
        "dpf": 0.5,
        "age": 30
    }

    if request.method == 'POST':
        try:
            # Collect form values
            form_data = {
                "pregnancies": int(request.form['pregnancies']),
                "glucose": int(request.form['glucose']),
                "bp": int(request.form['bp']),
                "skin": int(request.form['skin']),
                "insulin": int(request.form['insulin']),
                "bmi": float(request.form['bmi']),
                "dpf": float(request.form['dpf']),
                "age": int(request.form['age']),
            }

            input_data = np.array([[*form_data.values()]])
            input_scaled = scaler.transform(input_data)
            result = model.predict(input_scaled)[0]

            prediction = " The patient is likely to be diabetic." if result == 1 else " The patient is likely not diabetic."

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, data=form_data)


if __name__ == '__main__':
    app.run(debug=True)
