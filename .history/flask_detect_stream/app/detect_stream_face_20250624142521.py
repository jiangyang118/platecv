import cv2, time, os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# 加载模型
yolo = YOLO("models/yolo11n-seg.pt")
 
# 定义识别时间记录表
last_capture_time = {} 

# 主函数
def run_detection_face_food(video_source=0):
    cap = cv2.VideoCapture(video_source)
    os.makedirs("waste_captures", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = yolo.predict(source=frame, imgsz=640, verbose=False)[0]
        annotated = res.plot()
        waste_flag = False
        face_name = None

        # 计算浪费比
        masks = res.masks.data.cpu().numpy()
        plates = [(b.xyxy.cpu().numpy()[0], cls)
                  for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls==1]
        if masks.size > 0 and plates:
            food_mask = np.max(masks, axis=0).astype(np.uint8)
            food_area = food_mask.sum()
            x1, y1, x2, y2 = plates[0][0]
            plate_area = (y2 - y1) * (x2 - x1)
            ratio = food_area / plate_area
            waste_flag = (ratio > 0.25)
            cv2.putText(annotated, f"FoodRatio: {ratio:.1%}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 人脸检测与识别
        face_imgs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
        print(f"[INFO]sss Waste:{waste_flag}, Faces detected:{len(face_imgs)}")

        if waste_flag and face_imgs:
            face_img = face_imgs[0]["face"]
            face_name = recognize_face(face_img)

            # 如果是未知，不记录
            if face_name != "未知":
                now = time.time()
                last_time = last_capture_time.get(face_name, 0)
                if now - last_time > 60:
                    # 更新记录时间
                    last_capture_time[face_name] = now

                    # 标注姓名
                    cv2.putText(annotated, face_name, (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 保存图像
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"hahhawaste_captures/{face_name}_{ratio:.2f}_{timestamp}.jpg"
                    cv2.imwrite(fname, annotated)
                    print(f"[SAVE] {fname}")

        # 输出视频帧
        ret2, buf = cv2.imencode("2.jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()