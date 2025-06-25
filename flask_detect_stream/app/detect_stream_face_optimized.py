
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


yolo = YOLO("models/yolov8s-seg.pt") # 使用YOLOv8s-seg模型，确保模型路径正确
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
    针对每帧执行的完整业务逻辑，包括浪费判断、人脸识别和限流截图。
    :param res: YOLO预测结果
    :param frame: 原始帧
    :param annotated: 带注释的帧
    :param last_capture_time: 限流时间记录
    :param waste_threshold: 判断浪费的阈值
    :return: 更新后的 last_capture_time
    """

    # 安全判断：无分割结果
    if res.masks is None or res.masks.data is None:
        print("[WARNING] 当前帧无分割结果，跳过")
        return last_capture_time

    masks = res.masks.data.cpu().numpy()
    plates = [(b.xyxy.cpu().numpy()[0], cls)
              for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls == 1]

    waste_flag = False
    ratio = 0.0

    if masks.size > 0 and plates:
        frame_area = frame.shape[0] * frame.shape[1]
        waste_flag, ratio = is_waste_plate(res, frame_area, waste_threshold)
        draw_food_ratio_on_frame(annotated, plates[0][0], ratio)

    # 人脸提取
    face_imgs = DeepFace.extract_faces(frame, detector_backend="mtcnn", enforce_detection=False)
    print(f"[DEBUG] Waste: {waste_flag}, Faces detected: {len(face_imgs)}")

    # 封装调用：记录截图
    last_capture_time = process_face_and_capture(face_imgs, annotated, last_capture_time)

    return last_capture_time