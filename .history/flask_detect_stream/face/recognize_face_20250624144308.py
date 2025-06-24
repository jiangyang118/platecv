import os
from deepface import DeepFace
import cv2 

file_path = os.path.join(os.path.dirname(__file__), "name_map.json")
with open(file_path, "r", encoding="utf-8") as f:
    name_map = __import__('json').load(f)

db_path = os.path.join(os.path.dirname(__file__), "face_db")
if not os.path.exists(db_path):
    raise FileNotFoundError(f"[ERROR] 人脸库路径不存在: {db_path}")


# 人脸识别
def recg_face(face_img):
    dfs = DeepFace.find(img_path=face_img, db_path=db_path,
                       model_name='ArcFace', enforce_detection=False)
    print(f"[DEBUG] Face recognition results: {dfs}")
    if not dfs or dfs[0].empty:
        return "未知"
    eng = dfs[0].iloc[0]["identity"].split("/")[-2]
    print(f"[DEBUG] Recognized face: {eng}")
    return name_map.get(eng, "未知")

 
# 确保输出目录存在
os.makedirs("extracted_faces", exist_ok=True)

def save_face(face_imgs):
    for i, face in enumerate(face_imgs):
        face_rgb = face["face"]
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"extracted_faces/face_{i}.jpg", face_bgr)