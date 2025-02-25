import pickle
from flask import Flask, request, jsonify 
app = Flask( __name__ )
# Load the trained model
with open("model.pkl", "rb") as file: 
    model = pickle.load(file)
@app.route("/") 
def home():
    return "Welcome to the ML App!"
@app.route("/predict", methods=["POST"]) 
def predict():
    data = request.json
    prediction = model.predict([data["features"]]) 
    return jsonify({"prediction": prediction.tolist()})
if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=5000)