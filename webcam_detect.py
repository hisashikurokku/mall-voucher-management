from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import os
import numpy as np
from datetime import datetime
from server import send_plate_event
# import logging
# from ultralytics.utils import LOGGER

SHORT_MODEL_PATH = "short_plate_ds/runs/detect/train8/weights/best.pt"   # short plate YOLO model
LONG_MODEL_PATH  = "long_plate_ds/runs/detect/train2/weights/best.pt"  # long plate YOLO model
CONF_TH = 0.40
PADDING = 8

PADDING = 8
last_plate = None
last_seen_time = 0

short_model = YOLO(SHORT_MODEL_PATH)
long_model = YOLO(LONG_MODEL_PATH)
ocr = PaddleOCR(lang='en')

# logging.getLogger("ultralytics").setLevel(logging.ERROR)
# LOGGER.setLevel(logging.ERROR)
# logging.getLogger("ppocr").setLevel(logging.ERROR)

# def autocorrect(ch):
#     corrections = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
#     return corrections.get(ch, ch)

def detect_plate_bbox(img, try_long_first=False):
    if try_long_first:
        print("[DEBUG] Trying LONG model first based on aspect ratio")
    else:
        print("[DEBUG] Trying SHORT model first based on aspect ratio")

    primary   = long_model if try_long_first else short_model
    secondary = short_model if try_long_first else long_model

    results = primary(img)
    boxes   = results[0].boxes
    if len(boxes) > 0:
        boxes = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
        conf  = float(boxes[0].conf)
        if conf >= CONF_TH:
            print(f"[DEBUG] Plate detected by primary model: conf={conf:.3f}")
            return boxes[0]

    print("[DEBUG] Primary model failed → trying fallback model")
    results = secondary(img)
    boxes   = results[0].boxes
    if len(boxes) > 0:
        boxes = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
        conf  = float(boxes[0].conf)
        if conf >= CONF_TH:
            print(f"[DEBUG] Plate detected by fallback model: conf={conf:.3f}")
            return boxes[0]

    print("[DEBUG] Both models failed to detect plate")
    return None

def split_chars(poly, text):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    n = len(text)

    char_boxes = []
    for i, ch in enumerate(text):
        cx1 = xmin + (i * width / n)
        cx2 = xmin + ((i + 1) * width / n)
        box = [
            [cx1, ymin],
            [cx2, ymin],
            [cx2, ymax],
            [cx1, ymax]
        ]
        char_boxes.append((box, ch, 1.0))

    return char_boxes

def read_plate_from_crop(crop_img, min_row_gap_ratio=0.08):
    raw = ocr.predict(crop_img)

    detections = []
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        data = raw[0]
        polys  = data.get("dt_polys") or data.get("rec_polys") or []
        texts  = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        for i, poly in enumerate(polys):
            text  = texts[i] if i < len(texts) else ""
            score = float(scores[i]) if i < len(scores) else 1.0
            clean = ''.join([c for c in str(text).upper().strip() if c.isalnum() or c in "-."])
            if clean:
                detections.append((np.array(poly), clean, score))
    else:
        candidate = raw[0] if (isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list)) else raw
        for item in candidate:
            if len(item) == 2 and isinstance(item[1], tuple):
                box, (text, conf) = item
            elif len(item) >= 3:
                box, text, conf = item[0], item[1], item[2]
            else:
                continue
            clean = ''.join([c for c in str(text).upper().strip() if c.isalnum() or c in "-."])
            if clean:
                detections.append((np.array(box), clean, float(conf)))

    char_entries = []
    for poly, text, score in detections:
        text = text.strip()
        if len(text) == 1:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            xc = sum(xs) / len(xs)
            yc = sum(ys) / len(ys)
            char_entries.append((xc, yc, text, score, poly.tolist()))
        else:
            pieces = split_chars(poly, text)
            for box, ch, conf_e in pieces:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                xc = sum(xs) / len(xs)
                yc = sum(ys) / len(ys)
                char_entries.append((xc, yc, ch, score, box))

    if not char_entries:
        return "", [], detections

    y_coords = [e[1] for e in char_entries]
    y_min, y_max = min(y_coords), max(y_coords)
    h_crop = crop_img.shape[0]

    if (y_max - y_min) < max(2.0, min_row_gap_ratio * h_crop):
        char_entries.sort(key=lambda x: x[0])
        plate = "".join([c for _, _, c, _, _ in char_entries])
        row0  = [(x, y, ch) for x, y, ch, _, _ in char_entries]
        return plate, [row0], detections

    ys = [e[1] for e in char_entries]
    c1 = min(ys)
    c2 = max(ys)
    for _ in range(20):
        group1, group2 = [], []
        for yv in ys:
            if abs(yv - c1) <= abs(yv - c2):
                group1.append(yv)
            else:
                group2.append(yv)
        if not group1 or not group2:
            break
        new_c1 = sum(group1) / len(group1)
        new_c2 = sum(group2) / len(group2)
        if abs(new_c1 - c1) < 1e-3 and abs(new_c2 - c2) < 1e-3:
            break
        c1, c2 = new_c1, new_c2

    cluster1, cluster2 = [], []
    for e in char_entries:
        if abs(e[1] - c1) <= abs(e[1] - c2):
            cluster1.append(e)
        else:
            cluster2.append(e)

    mean1 = sum([e[1] for e in cluster1]) / len(cluster1) if cluster1 else float('inf')
    mean2 = sum([e[1] for e in cluster2]) / len(cluster2) if cluster2 else float('inf')
    if mean1 <= mean2:
        top_cluster, bottom_cluster = cluster1, cluster2
    else:
        top_cluster, bottom_cluster = cluster2, cluster1

    top_cluster.sort(key=lambda e: e[0])
    bottom_cluster.sort(key=lambda e: e[0])
    top_row    = "".join([e[2] for e in top_cluster])
    bottom_row = "".join([e[2] for e in bottom_cluster])

    rows_list = [
        [(e[0], e[1], e[2]) for e in top_cluster],
        [(e[0], e[1], e[2]) for e in bottom_cluster]
    ]

    return top_row + bottom_row, rows_list, detections

def crop_plate(full_img):
    H, W = full_img.shape[:2]
    aspect = H / W

    try_long_first = aspect < 0.5  # long plate shape → long model first

    best_box = detect_plate_bbox(full_img, try_long_first)

    if best_box is None:
        raise ValueError("No plate detected by either model.")

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    # Padding
    x1 = max(0, x1 - PADDING)
    y1 = max(0, y1 - PADDING)
    x2 = min(W - 1, x2 + PADDING)
    y2 = min(H - 1, y2 + PADDING)

    return full_img[y1:y2, x1:x2]

def format_vn_plate(top_row, bottom_row): # Format short plates
    """
    Format the Vietnamese license plate consisting of:
        Top row:    2 digits + 1–2 letters/digits (region + series)
        Bottom row: typically 5 digits, where last two form the fractional part

    Expected format -> XXAB-XXX.XX
    But some older plates omit the dot.

    This function:
        - concatenates top and bottom rows
        - inserts '-' after the top row
        - inserts '.' into the bottom row if needed
        - returns a cleaned final string
    """

    # 1. Remove any bad punctuation from OCR
    clean_top = ''.join([c for c in top_row if c.isalnum()])
    clean_bottom = ''.join([c for c in bottom_row if c.isalnum()])

    # 2. Insert dash between top and bottom
    combined = clean_top + "-" + clean_bottom

    # 3. If bottom row is 5 digits (normal), insert dot before last 2 digits
    if clean_bottom.isdigit():
        combined = f"{clean_top}-{clean_bottom[:3]}{clean_bottom[3:]}"

    # # 4. If bottom row is 4 digits (rare older plates)
    # elif len(clean_bottom) == 4 and clean_bottom.isdigit():
    #     combined = f"{clean_top}-{clean_bottom[:2]}{clean_bottom[2:]}"

    # 5. If bottom row already contains dot (e.g. OCR parsed it)
    # reformat it to XXX.XX
    elif "." in bottom_row:
        parts = clean_bottom.split(".")
        digits = "".join(parts)
        if len(digits) >= 5:
            combined = f"{clean_top}-{digits[:3]}{digits[3:5]}"

    return combined

def webcam_loop():
    global last_plate, last_sent_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Real-time recognition started — press Space to capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # Space
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            temp_path = f"capture_{timestamp}.jpg"

            cv2.imwrite(temp_path, frame)
            print("Frame captured → processing OCR pipeline")

            img = cv2.imread(temp_path)
            if img is None:
                print("Captured frame invalid")
                continue

            H, W = img.shape[:2]

            best_box = detect_plate_bbox(img)
            if best_box:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(W - 1, x2 + PADDING)
                y2 = min(H - 1, y2 + PADDING)

                crop = img[y1:y2, x1:x2]
                plate_text, rows_list, _ = read_plate_from_crop(crop)

                if rows_list:
                    if len(rows_list) == 1:
                        top_row = "".join(c for _,_,c in rows_list[0])
                        formatted = top_row.replace(".", "")
                    else:
                        top_row = "".join(c for _,_,c in rows_list[0])
                        bottom_row = "".join(c for _,_,c in rows_list[1])
                        formatted = format_vn_plate(top_row, bottom_row)

                    current_time = datetime.utcnow().timestamp()

                    if formatted != last_plate:
                        print("[DETECTED]", formatted)
                        send_plate_event(formatted, confidence=float(best_box.conf))
                        last_plate = formatted
                        last_sent_time = current_time
            else:
                print("No plate detected in captured frame")

            # Remove temporary image
            os.remove(temp_path)

        # Quit when pressing Q
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_loop()
