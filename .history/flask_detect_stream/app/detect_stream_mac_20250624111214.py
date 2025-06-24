# app/detect_stream_mac.py

import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# 加载 YOLO 模型
yolo = YOLO("models/yolo11n-seg.pt")  # 实例分割版，能输出 food mask

# 加载人脸库映射
with open('../../face/name_map.json', 'r', encoding='utf‑8') as f:
    name_map = __import__('json').load(f)
# 示例：{"jiangyang":"姜阳","lisi":"李四"}


# 加载人脸库
with open('../../face/face_embeddings.pkl','rb') as f:
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