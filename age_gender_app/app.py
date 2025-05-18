import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import urllib.request

# Load mô hình
model_age = load_model("best_age_model.keras")
model_gender = load_model("best_gender_model.keras")

# Khởi tạo MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 200))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Age & Gender Detector (MediaPipe)")

        # Căn giữa cửa sổ
        window_width = 500
        window_height = 650
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        root.resizable(False, False)

        # Tùy chỉnh giao diện
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Arial", 11), padding=6)
        style.configure("TLabel", font=("Arial", 12))
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))

        self.cap = None
        self.camera_running = False

        self.label = ttk.Label(root, text="🧠 Ứng dụng dự đoán tuổi & giới tính (MediaPipe)", style="Header.TLabel", anchor="center")
        self.label.pack(pady=10)

        self.image_label = ttk.Label(root)
        self.image_label.pack()

        self.result_label = ttk.Label(root, text="", anchor="center", wraplength=400, justify="center")
        self.result_label.pack(pady=10)

        ttk.Button(root, text="🎥 Bắt đầu Camera", command=self.start_camera).pack(pady=4)
        ttk.Button(root, text="⏸️ Tạm dừng Camera", command=self.stop_camera).pack(pady=4)
        ttk.Button(root, text="🖼️ Chọn ảnh từ máy", command=self.choose_image).pack(pady=4)
        ttk.Button(root, text="🌐 Chọn ảnh từ URL", command=self.choose_image_from_url).pack(pady=4)
        ttk.Button(root, text="❌ Thoát", command=self.quit_app).pack(pady=10)

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

    def choose_image_from_url(self):
        self.stop_camera()
        url = simpledialog.askstring("Nhập URL ảnh", "Vui lòng dán URL của ảnh:")
        if url:
            try:
                resp = urllib.request.urlopen(url)
                img_np = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                self.process_and_display(img)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải ảnh từ URL.\n{e}")

    def update_frame(self):
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.process_and_display(frame)
            self.root.after(20, self.update_frame)

    def process_and_display(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = face_detection.process(frame_rgb)

        frame_draw = frame.copy()
        results = []

        if results_mp.detections:
            h, w, _ = frame.shape
            for detection in results_mp.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)

                x, y = max(0, x), max(0, y)
                face_img = frame[y:y+bh, x:x+bw]

                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    continue

                input_face = preprocess(face_img)

                age = int(model_age.predict(input_face)[0][0])
                gender_score = model_gender.predict(input_face)[0][0]
                gender = "Female" if gender_score > 0.6 else "Male"

                results.append(f"👤 Tuổi: {age}, 🚻 Giới tính: {gender}")
                cv2.rectangle(frame_draw, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(frame_draw, f"{gender}, {age}", (x, y - 10),
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

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
