from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import uuid
import base64
import requests
from io import BytesIO
import mediapipe as mp

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

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def preprocess(image):
    image = cv2.resize(image, (200, 200))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def detect_faces(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    boxes = []
    if results.detections:
        ih, iw, _ = image.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            boxes.append((x, y, w, h))
    return boxes

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
                header, encoded = image_b64.split(",", 1)
                data = base64.b64decode(encoded)
                image_pil = Image.open(BytesIO(data)).convert("RGB")
                image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                filename = str(uuid.uuid4()) + ".jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(filepath, image)

        if image is not None:
            faces = detect_faces(image)

            for (x, y, w, h) in faces:
                face_img = image[y:y+h, x:x+w]
                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    continue
                input_face = preprocess(face_img)
                age = int(model_age.predict(input_face)[0][0])
                gender_score = model_gender.predict(input_face)[0][0]
                gender = "Female" if gender_score > 0.6 else "Male"
                results.append(f"Age: {age}, Gender: {gender}")
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"{gender}, {age}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(filepath, image)

    return render_template("index.html", results=results, filename=filename)

@app.route("/realtime", methods=["POST"])
def realtime():
    data = request.get_json()
    image_b64 = data.get("image_base64")

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    try:
        header, encoded = image_b64.split(",", 1)
        data = base64.b64decode(encoded)
        image_pil = Image.open(BytesIO(data)).convert("RGB")
        image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        faces = detect_faces(image)

        results = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue
            input_face = preprocess(face_img)
            age = int(model_age.predict(input_face)[0][0])
            gender_score = model_gender.predict(input_face)[0][0]
            gender = "Female" if gender_score > 0.6 else "Male"
            results.append({
                "label": f"Age: {age}, Gender: {gender}",
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_video():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue
            input_face = preprocess(face_img)
            age = int(model_age.predict(input_face)[0][0])
            gender_score = model_gender.predict(input_face)[0][0]
            gender = "Female" if gender_score > 0.6 else "Male"
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
