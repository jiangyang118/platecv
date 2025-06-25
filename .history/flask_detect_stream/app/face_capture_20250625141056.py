import time
import cv2
from face.recognize_face import recg_face_nums  # 请根据实际路径修改

# 限流状态应传入并由外部持久化管理
def process_face_and_capture(face_imgs, annotated, last_capture_time, output_dir="waste_captures"):
    if not face_imgs:
        return last_capture_time  # 无人脸

    face_img = face_imgs[0]["face"]

    try:
        face_name = recg_face_nums(face_img)
    except Exception as e:
        print(f"[ERROR] Face recognition failed: {e}")
        face_name = "未知"

    print(f"[INFO] Recognized face: {face_name}")
    now = time.time()
    print(f"[DEBUG] Current time: {now}, Last capture time: {last_capture_time.get(face_name, 0)}")

    # 限流：每人每60秒最多保存一次
    if face_name != "未知" and (face_name not in last_capture_time or now - last_capture_time[face_name] > 60):
        last_capture_time[face_name] = now
        color = (0, 0, 255)
        cv2.putText(annotated, face_name, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{output_dir}/{face_name}_{timestamp}.jpg"
        # fname = f"waste_captures/{face_name}_{ratio:.2f}_{timestamp}.jpg"
        cv2.imwrite(fname, annotated)
        print(f"[SAVE] {fname}")

    return last_capture_time