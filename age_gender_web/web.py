# app.py
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import base64
import requests
from io import BytesIO

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_age = load_model(os.path.join(BASE_DIR, "..", "model_age.h5"))
model_gender = load_model(os.path.join(BASE_DIR, "..", "model_gender.h5"))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess(image):
    image = cv2.resize(image, (200, 200))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    filename = None

    if request.method == "POST":
        mode = request.form.get("mode")
        image = None

        if mode == "upload" and 'image' in request.files:
            file = request.files['image']
            if file.filename:
                filename = str(uuid.uuid4()) + ".jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image = cv2.imread(filepath)

        elif mode == "url":
            image_url = request.form.get("image_url")
            if image_url:
                try:
                    response = requests.get(image_url)
                    image_pil = Image.open(BytesIO(response.content)).convert("RGB")
                    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                    filename = str(uuid.uuid4()) + ".jpg"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    cv2.imwrite(filepath, image)
                except:
                    image = None

        elif mode == "camera":
            image_b64 = request.form.get("image_base64")
            if image_b64:
                header, encoded = image_b64.split(",")
                data = base64.b64decode(encoded)
                image_pil = Image.open(BytesIO(data)).convert("RGB")
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                filename = str(uuid.uuid4()) + ".jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(filepath, image)

        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    continue
                input_face = preprocess(face_img)
                age = int(model_age.predict(input_face)[0][0])
                gender_score = model_gender.predict(input_face)[0][0]
                gender = "Male" if gender_score > 0.5 else "Female"
                results.append(f"Age: {age}, Gender: {gender}")
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{gender}, {age}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(filepath, image)

    return render_template("index.html", results=results, filename=filename)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
