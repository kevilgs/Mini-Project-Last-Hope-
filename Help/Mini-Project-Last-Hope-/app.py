from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/logistic_regression_model.pkl')

# Load the symptom encoder
symptom_encoder = joblib.load('model/symptom_encoder.pkl')

# Load the disease encoder
disease_encoder = joblib.load('model/disease_encoder.pkl')

# Load the column names used during training
with open('model/columns.pkl', 'rb') as f:
    columns = joblib.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']  # Get symptoms from form
    symptoms_list = [symptom.strip() for symptom in symptoms.split(',')]  # Split and strip symptoms
    # Encode symptoms as done during training
    symptom_encoded = symptom_encoder.transform(symptoms_list)
    symptom_df = pd.DataFrame(symptom_encoded, columns=['Symptom_Encoded'])  # Create DataFrame with encoded symptoms
    symptom_df = pd.get_dummies(symptom_df['Symptom_Encoded'])  # Create dummy variables

    # Align the dummy variables with the training columns
    symptom_df = symptom_df.reindex(columns=columns, fill_value=0)

    # Make prediction
    probabilities = model.predict_proba(symptom_df)[0]  # Get probabilities for each class
    top_indices = np.argsort(probabilities)[-3:][::-1]  # Get indices of top 3 predictions
    top_diseases = disease_encoder.inverse_transform(top_indices)  # Map indices to disease names
    top_probabilities = probabilities[top_indices]  # Get probabilities of top predictions

    results = list(zip(top_diseases, top_probabilities))  # Combine diseases and probabilities

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
