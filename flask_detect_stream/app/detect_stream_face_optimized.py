
import cv2, time, os
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from face.recognize_face import recg_face  # 假设你有一个人脸识别模块
from face.recognize_face import save_face  # 假设你有一个保存人脸图片的模块
from face.recognize_face import recg_face_nums
from app.waste_detector import is_waste_plate, draw_food_ratio_on_frame 
from app.face_capture import process_face_and_capture  # 根据你的路径调整

# 模型加载
# yolo = YOLO("models/yolo11n-seg.pt")  # 实例分割模型


# yolo = YOLO("models/yolov8s-seg.pt") # 使用YOLOv8s-seg模型，确保模型路径正确
yolo = YOLO("models/yolov8n-seg.pt") 

# yolo = YOLO("models/yolo11n.pt")  # 替换为实际模型路径
# 记录每个人上次截图时间
last_capture_time = {}

def run_detection_face_food2(video_source=0):
    global last_capture_time
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

        process_frame_logic(res, frame, annotated, last_capture_time) 

        # 视频流返回
        ret2, buf = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release() 

def process_frame_logic(res, frame, annotated, last_capture_time, waste_threshold=0.25):
    """
    优化后业务逻辑：先检测人脸，有人才判断是否有浪费，避免空耗资源。
    """
    # 🔍 Step 1: 提取人脸
    try:
        face_imgs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
    except Exception as e:
        print(f"[ERROR] Face extraction failed: {e}")
        face_imgs = []
 
 
    if not face_imgs:
        print("[DEBUG] 无人脸，跳过本帧")
        return last_capture_time  # 没有人脸，跳过后续计算

    print(f"[DEBUG] Detected Faces: {len(face_imgs)}")

    # 🧠 Step 2: 判断是否存在浪费（先确保有分割结果）
    if res.masks is None or res.masks.data is None:
        print("[WARNING] 当前帧无分割结果，跳过浪费判断")
        return last_capture_time

    masks = res.masks.data.cpu().numpy()
    plates = [(b.xyxy.cpu().numpy()[0], cls)
              for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls == 1]

    if not (masks.size > 0 and plates):
        print("[DEBUG] 无餐盘或食物，跳过浪费判断")
        return last_capture_time

    frame_area = frame.shape[0] * frame.shape[1]
    waste_flag, ratio = is_waste_plate(res, frame_area, waste_threshold)

    if not waste_flag:
        print("[DEBUG] 检测到有人，但无浪费行为，跳过截图")
        return last_capture_time

    # 🎯 Step 3: 有人 + 有浪费 → 标注 + 限流截图
    draw_food_ratio_on_frame(annotated, plates[0][0], ratio)
    last_capture_time = process_face_and_capture(face_imgs, annotated, last_capture_time)

    return last_capture_time
