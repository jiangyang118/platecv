from ultralytics import YOLO
import cv2

def main():
    model = YOLO("models/yolo11n.pt")
    source = 0  # 本地摄像头；也可使用 RTSP/HLS URL
    for result in model.predict(source, stream=True):
        frame = result.orig_img
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("YOLO11 Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# 安装 Python 和依赖
# brew install python3
# python3 -m venv yolov11
# source yolov11/bin/activate

# pip install ultralytics opencv-python inference
# python detect_stream_mac.py