import cv2
from deepface import DeepFace

from app.detect_stream_face_optimized import run_detection_face_food2


with open('./face/name_map.json', 'r', encoding='utf‑8') as f:
    name_map = __import__('json').load(f)

# 人脸识别
def recognize_face(face_img):
    dfs = DeepFace.find(img_path=face_img, db_path='./face_db',
                       model_name='ArcFace', enforce_detection=False)
    print(f"[DEBUG] Face recognition results: {dfs}")
    if not dfs or dfs[0].empty:
        return "未知"
    eng = dfs[0].iloc[0]["identity"].split("/")[-2]
    return name_map.get(eng, "未知")
