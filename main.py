
import cv2
from collections import deque
from utils.detection import detect_face_and_plate
from utils.segmentation import calc_waste_ratio
from utils.recorder import save_snapshot, save_video_clip
from utils.holding import is_holding

cap = cv2.VideoCapture(0)  # 替换为 RTSP 地址也可
buffer = deque(maxlen=300)  # 10 秒缓存（假设 30fps）

while True:
    ret, frame = cap.read()
    if not ret:
        break
    buffer.append(frame)

    face_box, plate_box = detect_face_and_plate(frame)
    # print("face_box")
    # print(face_box)
    # print("plate_box")
    # print(plate_box)
    if face_box and plate_box and is_holding(face_box, plate_box):
        plate_crop = frame[plate_box[1]:plate_box[3], plate_box[0]:plate_box[2]]
        waste_ratio = calc_waste_ratio(plate_crop)
        if waste_ratio > 0.15:
            print(f"[WASTE] Detected: {waste_ratio:.2%}")
            save_snapshot(frame, waste_ratio)
            save_video_clip(buffer)

    # face_box, plate_box = detect_face_and_plate(frame)
    # if plate_box:
    #     x1, y1, x2, y2 = plate_box
    #     plate_crop = frame[y1:y2, x1:x2]
    #     waste_ratio = calc_waste_ratio(plate_crop)

    #     if waste_ratio > 0.15:
            
    #         save_snapshot(frame, waste_ratio)
    #         save_video_clip(buffer)

    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
