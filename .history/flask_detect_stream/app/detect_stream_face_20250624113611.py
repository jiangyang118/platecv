import cv2, time, os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import json
from collections import defaultdict
from datetime import datetime

# 加载 YOLO 分割模型
yolo = YOLO("models/yolo11n-seg.pt")

# 加载人员映射字典
with open('../../face/name_map.json', 'r', encoding='utf‑8') as f:
    name_map = json.load(f)

# 识别频率控制（避免频繁截图）
last_capture_time = defaultdict(lambda: 0)

# DeepFace 识别函数（输入：face图像，输出：姓名）
def recognize_face(face_img):
    try:
        result = DeepFace.find(img_path=face_img, db_path='../../face/face_db',
                               model_name='ArcFace', enforce_detection=False, detector_backend='opencv')
        if result.empty:
            return "未知"
        folder_name = os.path.basename(os.path.dirname(result.iloc[0]["identity"]))
        return name_map.get(folder_name, "未知")
    except Exception as e:
        print(f"[ERROR] DeepFace failed: {e}")
        return "未知"

def run_detection_face_food(video_source=0):
    cap = cv2.VideoCapture(video_source)
    os.makedirs("waste_captures", exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = yolo.predict(source=frame, imgsz=640, verbose=False)[0]
        annotated = res.plot()
        waste_flag = False
        face_name = "未知"

        # 计算浪费比例
        try:
            masks = res.masks.data.cpu().numpy() if res.masks else np.array([])
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            plates = [box for box, cls in zip(boxes, classes) if int(cls) == 1]

            if masks.size > 0 and len(plates) > 0:
                food_mask = np.max(masks, axis=0).astype(np.uint8)
                food_area = food_mask.sum()
                x1, y1, x2, y2 = plates[0]
                plate_area = (x2 - x1) * (y2 - y1)
                if plate_area > 0:
                    ratio = food_area / plate_area
                    waste_flag = ratio > 0.25
                    cv2.putText(annotated, f"FoodRatio: {ratio:.1%}",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        except Exception as e:
            print(f"[ERROR] Waste calc failed: {e}")

        # 人脸识别
        face_imgs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
        has_face = len(face_imgs) > 0
        if has_face:
            face_img = face_imgs[0]["face"]
            face_name = recognize_face(face_img).strip()
            if not face_name:
                face_name = "未知"
            color = (0, 0, 255) if waste_flag else (0, 255, 0)
            cv2.putText(annotated, face_name, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 满足条件且频率控制通过才保存截图
        if waste_flag and has_face:
            now = time.time()
            if now - last_capture_time[face_name] > 60:  # 每人每分钟最多1张
                last_capture_time[face_name] = now
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
                # 替换非法字符用于文件名安全（防止不同 face_name 被识别为不同人）
                safe_name = "".join(c for c in face_name if c.isalnum() or c in "_-").lower()
                filename = f"waste_captures/{safe_name}_{ratio:.2f}_{timestamp}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"[DEBUG] face={safe_name}, now={now}, last={last_capture_time[safe_name]}")

        # MJPEG 推流
        ret2, buf = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()