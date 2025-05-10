import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load mô hình
model_age = load_model("model_gender.h5")
model_gender = load_model("model_gender.h5")

# Phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ✅ BGR -> RGB
    image = cv2.resize(image, (200, 200))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Age & Gender Detector")
        self.cap = None
        self.camera_running = False

        self.label = Label(root, text="🧠 Ứng dụng dự đoán tuổi & giới tính", font=("Arial", 14))
        self.label.pack(pady=10)

        self.image_label = Label(root)
        self.image_label.pack()

        self.result_label = Label(root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

        Button(root, text="🎥 Bắt đầu Camera", command=self.start_camera).pack(pady=2)
        Button(root, text="⏸️ Tạm dừng Camera", command=self.stop_camera).pack(pady=2)
        Button(root, text="🖼️ Chọn ảnh từ máy", command=self.choose_image).pack(pady=2)
        Button(root, text="❌ Thoát", command=self.quit_app).pack(pady=10)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.camera_running = True
        self.update_frame()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
        self.camera_running = False
        self.cap = None

    def choose_image(self):
        self.stop_camera()
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            img = cv2.imread(path)
            self.process_and_display(img)

    def update_frame(self):
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display(frame)
            self.root.after(20, self.update_frame)

    def process_and_display(self, frame):
        frame_draw = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        results = []
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Bỏ qua khuôn mặt quá nhỏ
            if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                continue

            input_face = preprocess(face_img)

            age = int(model_age.predict(input_face)[0][0])
            gender_score = model_gender.predict(input_face)[0][0]
            gender = "Male" if gender_score > 0.5 else "Female"

            results.append(f"👤 Tuổi: {age}, 🚻 Giới tính: {gender}")
            cv2.rectangle(frame_draw, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_draw, f"{gender}, {age}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        img_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        self.result_label.configure(text="\n".join(results))

    def quit_app(self):
        self.stop_camera()
        self.root.destroy()

# Khởi chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
