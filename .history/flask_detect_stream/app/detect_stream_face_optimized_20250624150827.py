
import cv2, time, os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from face.recognize_face import recg_face  # 假设你有一个人脸识别模块
from face.recognize_face import save_face  # 假设你有一个保存人脸图片的模块
from face.recognize_face import recg_face_nums
from waste_detector import is_waste_plate, draw_food_ratio_on_frame 
from face_capture import process_face_and_capture  # 根据你的路径调整

# 模型加载
yolo = YOLO("models/yolo11n-seg.pt")  # 实例分割模型

# 记录每个人上次截图时间
last_capture_time = {}

def run_detection_face_food2(video_source=0):
    cap = cv2.VideoCapture(video_source)
    os.makedirs("waste_captures", exist_ok=True)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 控制处理频率
        if frame_count % 3 != 0:
            continue

        res = yolo.predict(source=frame, imgsz=640, verbose=False)[0]
        annotated = res.plot()
        waste_flag = False
        face_name = "未知"

        masks = res.masks.data.cpu().numpy()
        plates = [(b.xyxy.cpu().numpy()[0], cls)
                for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls == 1]

        if masks.size > 0 and plates:
            waste_flag, ratio = is_waste_plate(res, frame_area=frame.shape[0] * frame.shape[1], waste_threshold=0.25)
            draw_food_ratio_on_frame(annotated, plates[0][0], ratio)


        # 提取人脸
        face_imgs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
        # save_face(face_imgs)  # 保存提取的人脸图片
        print(f"[DEBUG] Waste: {waste_flag}, Faces detected: {len(face_imgs)}")

        # if waste_flag and face_imgs:
        # 替换为封装后的调用
        last_capture_time = process_face_and_capture(face_imgs, annotated, last_capture_time)
 

        # 视频流返回
        ret2, buf = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()
