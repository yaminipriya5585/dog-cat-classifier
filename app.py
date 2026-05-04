from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Open image
    img = Image.open(file).convert("RGB")
    img = img.resize((150, 150))

    # Convert to array
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)

    # Result
    result = "Dog 🐶" if pred[0][0] > 0.5 else "Cat 🐱"

    return render_template("index.html", prediction=result)

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)