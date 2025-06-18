
import cv2
import time
from datetime import datetime

def save_snapshot(frame, ratio):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/waste_{ratio:.2f}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[INFO] Snapshot saved: {filename}")

def save_video_clip(buffer, fps=30):
    if not buffer:
        return
    height, width, _ = buffer[0].shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/waste_clip_{timestamp}.mp4"

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in buffer:
        out.write(frame)
    out.release()
    print(f"[INFO] 10s clip saved: {filename}")
