
import cv2
import numpy as np

def calc_waste_ratio(crop_img):
    # 将图像转 HSV 过滤食物颜色范围（仅示例）
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 40])
    upper = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    waste_area = np.sum(mask > 0)
    total_area = crop_img.shape[0] * crop_img.shape[1]
    return waste_area / total_area
