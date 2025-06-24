import cv2
from deepface import DeepFace
import os

# 模拟 name_map.json
name_map = {
    "person1": "张三",
    "person2": "李四"
}

# 被测方法
def recognize_face(face_img_path):
    df = DeepFace.find(img_path=face_img_path, db_path='../../face/face_db',
                       model_name='ArcFace', enforce_detection=False)
    print(df)  # 输出调试信息
    if df.empty:
        return "未知"
    eng = df.iloc[0]["identity"].split("/")[-2]
    return name_map.get(eng, "未知")

# 测试入口
if __name__ == "__main__":
    # 替换为你本地的一张待识别人脸照片路径
    test_face_path = "test_face.jpg"

    if not os.path.exists(test_face_path):
        print(f"[ERROR] 测试图片 {test_face_path} 不存在，请确认路径正确")
    else:
        result = recognize_face(test_face_path)
        print(f"[RESULT] 识别结果：{result}")