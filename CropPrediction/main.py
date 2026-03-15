from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        prediction = model.predict(final_features)

        return render_template("index.html",
                               prediction_text=f"Predicted Crop: {prediction[0]}")
    except:
        return render_template("index.html",
                               prediction_text="⚠️ Please enter valid numbers")

if __name__ == "__main__":
    app.run(debug=True)