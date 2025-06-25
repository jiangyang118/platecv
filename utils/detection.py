
from ultralytics import YOLO

# 加载模型（路径请根据实际情况替换）
face_model = YOLO("models/yolo11n.pt")
plate_model = YOLO("models/custom_plate.pt")

def detect_face_and_plate(frame):
    face_result = face_model.predict(frame, verbose=False)[0]
    plate_result = plate_model.predict(frame, verbose=False)[0]
    # print("face_box")
    # print(face_box)
    # print("plate_result")
    # print(plate_result)
    face_box = None
    plate_box = None

    if face_result.boxes:
        face = face_result.boxes.xyxy[0].cpu().numpy().astype(int)
        face_box = face.tolist()

    if plate_result.boxes:
        plate = plate_result.boxes.xyxy[0].cpu().numpy().astype(int)
        plate_box = plate.tolist()

    return face_box, plate_box

