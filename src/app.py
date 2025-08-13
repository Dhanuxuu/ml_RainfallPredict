from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("notebooks\Rainfall_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    feature1 = float(data["feature1"])
    feature2 = float(data["feature2"])
    feature3 = float(data["feature3"])
    feature4 = float(data["feature4"])
    
    prediction = model.predict([[feature1, feature2,feature3, feature4]])
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)

