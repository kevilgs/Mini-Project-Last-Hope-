# Medical Diagnosis System

This project is a web-based medical diagnosis system that predicts diseases based on user-entered symptoms. The system uses a machine learning model trained on a dataset of diseases and symptoms.

## Features

- Enter symptoms manually or select from a dropdown list.
- Predict the top 3 possible diseases based on the entered symptoms.
- Display the probability of each predicted disease.

## Installation

1. Clone the repository/Download the zip file(recommended)
    ```sh
    git clone https://github.com/yourusername/medical-diagnosis-system.git
    cd medical-diagnosis-system
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    # source .venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    pip install flask pandas imblearn sckilit-learn
    ```

4. Place your datasets in the appropriate directory:
    - `dis_sym_dataset_comb.csv`
    - `dis_sym_dataset_norm.csv`
    - `symptoms_file.csv`
      Dont forget to change the path in the train_model.py

5. Train the model:
    ```sh
    python train_model.py
    ```

6. Run the Flask application:
    ```sh
    python app.py
    ```

## Usage

1. Open your web browser and go to `http://127.0.0.1:5000/`.
2. Enter symptoms manually or select symptoms from the dropdown list.
3. Click on "Predict Disease" to get the top 3 possible diseases along with their probabilities.

## Project Structure
