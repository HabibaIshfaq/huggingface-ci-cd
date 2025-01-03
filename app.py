from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Correctly specify the model name
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.json
    text = data.get("text", "")
    
    # If no text is provided, return an error response
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Use the model to make a prediction
    result = model(text)
    
    # Return the result as JSON
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False)


