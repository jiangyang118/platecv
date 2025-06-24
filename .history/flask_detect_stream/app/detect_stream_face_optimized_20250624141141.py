
import cv2, time, os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from recognize_face import recognize_face  # 假设你有一个人脸识别模块

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

        # 计算餐盘浪费比例
        masks = res.masks.data.cpu().numpy()
        plates = [(b.xyxy.cpu().numpy()[0], cls)
                  for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls == 1]
        if masks.size > 0 and plates:
            food_mask = np.max(masks, axis=0).astype(np.uint8)
            food_area = food_mask.sum()
            x1, y1, x2, y2 = plates[0][0]
            plate_area = (y2 - y1) * (x2 - x1)
            ratio = food_area / plate_area
            waste_flag = ratio > 0.25
            cv2.putText(annotated, f"FoodRatio: {ratio:.1%}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 提取人脸
        face_imgs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)

        print(f"[DEBUG] Waste: {waste_flag}, Faces detected: {len(face_imgs)}")

        # if waste_flag and face_imgs:
        if face_imgs:
            face_img = face_imgs[0]["face"]
            try:
                face_name = recognize_face(face_img)
            except Exception as e:
                print(f"[ERROR] Face recognition failed: {e}")
                face_name = "未知" 
            print(f"[INFO] Recognized face: {face_name}")
            # 限流每人每分钟最多一次
            now = time.time()
            print(f"[DEBUG] Current time: {now}, Last capture time: {last_capture_time.get(face_name, 0)}")
            if face_name not in last_capture_time or now - last_capture_time[face_name] > 60:
                last_capture_time[face_name] = now
                color = (0, 0, 255)
                cv2.putText(annotated, face_name, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                fname = f"waste_captures/{face_name}_{timestamp}.jpg"
                # fname = f"waste_captures/{face_name}_{ratio:.2f}_{timestamp}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"[SAVE] {fname}")

        # 视频流返回
        ret2, buf = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()
