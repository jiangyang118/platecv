# app/detect_stream_mac.py

from ultralytics import YOLO
import cv2

def run_detection(video_source=0):
    model = YOLO("models/yolo11n.pt")  # 替换成你的模型路径
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()