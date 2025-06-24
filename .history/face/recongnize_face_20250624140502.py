

# 人脸识别
def recognize_face(face_img):
    dfs = DeepFace.find(img_path=face_img, db_path='./face_db',
                       model_name='ArcFace', enforce_detection=False)
    print(f"[DEBUG] Face recognition results: {dfs}")
    if not dfs or dfs[0].empty:
        return "未知"
    eng = dfs[0].iloc[0]["identity"].split("/")[-2]
    return name_map.get(eng, "未知")
