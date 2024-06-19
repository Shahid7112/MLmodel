from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_classifier.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    # Make prediction
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
