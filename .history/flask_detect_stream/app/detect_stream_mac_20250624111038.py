# app/detect_stream_mac.py

from ultralytics import YOLO
import cv2
from deepface import DeepFace
import pickle, numpy as np, cv2


# 加载人脸库
with open('face_embeddings.pkl','rb') as f:
    face_db = pickle.load(f)
    
def run_detection(video_source=0):
    model = YOLO("models/yolo11n.pt")  # 替换为实际模型路径
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # 将处理后图像编码为 JPEG 格式并输出为流
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()