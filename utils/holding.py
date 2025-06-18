

def is_holding(face_box, plate_box):
    fx1, _, fx2, _ = face_box
    px1, _, px2, _ = plate_box
    overlap = min(fx2, px2) - max(fx1, px1)
    return overlap > 0