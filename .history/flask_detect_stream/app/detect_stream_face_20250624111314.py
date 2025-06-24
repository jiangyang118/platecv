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

# 封装 DeepFace.find 的识别函数
def recognize_face(img):
    df = DeepFace.find(img_path=img, db_path="../../face/face_db/",
                       model_name="ArcFace", enforce_detection=False)
    if df.empty:
        return "未知"
    eng = df.iloc[0]["identity"].split("/")[-2]
    return name_map.get(eng, "未知")

def run_detection(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = yolo.predict(source=frame, imgsz=640, verbose=False)[0]
        annotated = res.plot()
        waste_flag = False
        # 计算 food mask 面积比例
        masks = res.masks.data.cpu().numpy()
        if masks.size > 0 and res.boxes:
            food_mask = np.max(masks, axis=0).astype(np.uint8)
            food_area = food_mask.sum()
            # 获取盘子 bbox 面积
            plates = [(b.xyxy.cpu().numpy()[0], cls) for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls==1]
            if plates:
                x1,y1,x2,y2 = plates[0][0]
                plate_area = (y2-y1)*(x2-x1)
                waste_flag = (food_area/plate_area) > 0.25
                cv2.putText(annotated, f"FoodRatio: {food_area/plate_area:.2%}",
                            (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

        # 若检测到人脸，进行识别并标名
        face_imgs = DeepFace.extract_faces(frame, detector_backend="opencv", enforce_detection=False)
        if face_imgs:
            face = face_imgs[0]["face"]
            name = recognize_face(face)
            color = (0,0,255) if waste_flag else (0,255,0)
            cv2.putText(annotated, f"{name}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2)

        # 输出 MJPEG 流
        ret2, buf = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()