from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "age_gender_api/age_gender_web/static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Load models
model_age = load_model("age_gender_api/model_age.h5")
model_gender = load_model("age_gender_api/model_gender.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess(image):
    image = cv2.resize(image, (200, 200))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            results = []
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    continue

                input_face = preprocess(face_img)
                age = int(model_age.predict(input_face)[0][0])
                gender_score = model_gender.predict(input_face)[0][0]
                gender = "Male" if gender_score > 0.5 else "Female"

                results.append(f"Age: {age}, Gender: {gender}")
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{gender}, {age}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save annotated image
            cv2.imwrite(filepath, img)

            return render_template("index.html", filename=filename, results=results)

    return render_template("index.html")

@app.route("/age_gender_api/age_gender_web/static/uploads/<filename>")
def send_file(filename):
    return redirect(url_for('static', filename=f"uploads/{filename}"))

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)