import cv2
import time
from ultralytics import YOLO
from face.recognize_face import extract_faces
from face.capture_utils import process_face_and_capture
from app.waste_detector import is_waste_plate, draw_food_ratio_on_frame

# 初始化模型和状态
model = YOLO("models/yolo11n.pt")  # 轻量级模型
last_capture_time = {}

def run_detection_face_food(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        success, frame = cap.read()
        if not success:
            break

        res = model(frame)[0]  # YOLO 推理
        annotated = res.plot()  # 带注释的图像

        # 餐盘浪费检测
        is_waste, ratio, plate_coords = is_waste_plate(res, frame.shape)
        if is_waste:
            draw_food_ratio_on_frame(annotated, plate_coords, ratio)

        # 提取人脸并进行限流截图
        face_imgs = extract_faces(frame)
        global last_capture_time
        last_capture_time = process_face_and_capture(face_imgs, annotated, last_capture_time)

        # 编码后输出帧
        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()
