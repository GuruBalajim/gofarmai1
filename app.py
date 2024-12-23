from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os

app = Flask(__name__)
app.secret_key = "secret_key"

# Initialize Hugging Face model for text generation
# Using Hugging Face GPT-like open models such as OPT or GPT-J



# Load pre-trained models
cnn_model = load_model('CNN.model')  # Ensure the path to your CNN model
ml_model = pickle.load(open('model.sav', 'rb'))

# Dataset directory and class names
data_dir = "dataset"
class_names = os.listdir(data_dir)

# Function to generate AI advice using Hugging Face GPT-like models

@app.route("/chatbox")
def chatbox():
    return render_template("chatbox.html")
@app.route("/base")
def base():
    return render_template("base.html")
# Create base.html
@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/community")
def community():
    return render_template("community.html")  # Create community.html

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pest_detection", methods=["GET", "POST"])
def pest_detection():
    if request.method == "POST":
        try:
            # Get input data
            n = int(request.form['N'])
            p = int(request.form['P'])
            k = int(request.form['K'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['pH'])
            rainfall = float(request.form['Rainfall'])
        except ValueError:
            return render_template("pest_detection.html", error="Invalid input data")

        # File upload
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("pest_detection.html", error="Please upload an image")

        file = request.files["image"]
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Process image
        img = cv2.imread(filepath)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))
        img_expanded = np.expand_dims(img, axis=0)
        img_normalized = img_expanded / 255.0

        # Predict pest
        predictions = cnn_model.predict(img_normalized)
        pest_result = class_names[np.argmax(predictions)]

        # Predict crop
        crop_inputs = [[n, p, k, temp, humidity, ph, rainfall]]
        crop_prediction = ml_model.predict(crop_inputs)[0]

        return render_template(
            "pest_detection.html",
            pest_result=pest_result,
            crop_prediction=crop_prediction,
            image_url=filepath,
        )
    return render_template("pest_detection.html")
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.form["message"]
        prompt = f"You: {user_message}\nAI:"
        response = text_generator(
            prompt, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2
        )
        chat_response = response[0]["generated_text"]
        return ''.join(char for char in chat_response if char.isprintable())
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True) 