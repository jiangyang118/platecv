import numpy as np
import cv2


def calculate_plate_area(bbox):
    x1, y1, x2, y2 = bbox
    return (y2 - y1) * (x2 - x1)


def calculate_food_area(masks):
    if masks.size == 0:
        return 0
    food_mask = np.max(masks, axis=0).astype(np.uint8)
    return food_mask.sum(), food_mask


def calculate_food_plate_ratio(food_area, plate_area):
    if plate_area == 0:
        return 0.0
    return food_area / plate_area


def is_waste_plate(res, frame_area, waste_threshold=0.15):
    masks = res.masks.data.cpu().numpy()
    plates = [(b.xyxy.cpu().numpy()[0], cls)
              for b, cls in zip(res.boxes, res.boxes.cls.cpu().numpy()) if cls == 1]
    
    if masks.size == 0 or not plates:
        return False, 0.0

    food_area, food_mask = calculate_food_area(masks)
    bbox = plates[0][0]
    plate_area = calculate_plate_area(bbox)
    ratio = calculate_food_plate_ratio(food_area, plate_area)
    
    return ratio > waste_threshold, ratio


def draw_food_ratio_on_frame(annotated, bbox, ratio):
    x1, y1, x2, y2 = bbox
    cv2.putText(
        annotated,
        f"FoodRatio: {ratio:.1%}",
        (int(x1), int(y1) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )