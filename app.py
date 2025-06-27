from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
scaler = joblib.load('scaler_final.pkl')
model = joblib.load('model_final.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        at = float(request.form['AT'])
        ap = float(request.form['AP'])
        ah = float(request.form['AH'])
        afdp = float(request.form['AFDP'])
        gtep = float(request.form['GTEP'])
        tit = float(request.form['TIT'])
        tat = float(request.form['TAT'])

        input_data = np.array([[at, ap, ah, afdp, gtep, tit, tat]])
        scaled_input = scaler.transform(input_data)
        predicted_tey = model.predict(scaled_input)[0]

        return render_template('index.html', prediction=f"{predicted_tey:.2f} MWH")

    except Exception as e:
        return render_template('index.html', prediction=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
