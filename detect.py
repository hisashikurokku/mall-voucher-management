from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np
from server import send_plate_event

# --------------------------
# CONFIG
# --------------------------
SHORT_MODEL_PATH = "short_plate_ds/runs/detect/train8/weights/best.pt"   # short plate YOLO model
LONG_MODEL_PATH  = "long_plate_ds/runs/detect/train2/weights/best.pt"  # long plate YOLO model
INPUT_IMAGE = "path/to/test/image"
CONF_TH = 0.40
PADDING = 8

# --------------------------
# Init models
# --------------------------
short_model = YOLO(SHORT_MODEL_PATH)
long_model = YOLO(LONG_MODEL_PATH)
ocr = PaddleOCR(lang='en', det_model_dir=None, rec_model_dir=None)

# def autocorrect(ch):
#     corrections = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
#     return corrections.get(ch, ch)

def detect_plate_bbox(img, try_long_first=False):
    """Return best YOLO bbox or None, also print debug info."""

    if try_long_first:
        print("[DEBUG] Trying LONG model first based on aspect ratio")
    else:
        print("[DEBUG] Trying SHORT model first based on aspect ratio")

    primary = long_model if try_long_first else short_model
    secondary = short_model if try_long_first else long_model

    # Try primary model
    results = primary(img)
    boxes = results[0].boxes
    if len(boxes) > 0:
        boxes = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
        conf = float(boxes[0].conf)
        if conf >= CONF_TH:
            model_used = "LONG" if try_long_first else "SHORT"
            print(f"[DEBUG] {model_used} model detected plate: conf={conf:.3f}")
            return boxes[0]

    # Try fallback model
    print("[DEBUG] Primary model failed → trying fallback model")
    results = secondary(img)
    boxes = results[0].boxes
    if len(boxes) > 0:
        boxes = sorted(boxes, key=lambda b: float(b.conf), reverse=True)
        conf = float(boxes[0].conf)
        if conf >= CONF_TH:
            model_used = "SHORT" if try_long_first else "LONG"
            print(f"[DEBUG] {model_used} model fallback detection: conf={conf:.3f}")
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
    """
    crop_img: BGR image (numpy array) containing mostly the plate
    Returns:
        plate_str (merged string) ,
        rows_list: [ [ (x_center, y_center, char), ... ] top-first ],
        raw_detections: list of (poly, text, score)
    Behavior:
        - Splits multi-char detections into per-char boxes (as before)
        - Computes (x_center, y_center) per char
        - If vertical distribution suggests two rows, separates them using simple 1D k-means
        - Sorts left->right inside each row
    """
    # Run OCR (keeps your existing paddlex or paddle outputs handling)
    raw = ocr.predict(crop_img)

    # Normalize raw -> list of (poly, text, score)
    detections = []
    # Support paddlex dict output
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        data = raw[0]
        polys = data.get("dt_polys") or data.get("rec_polys") or []
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        for i, poly in enumerate(polys):
            text = texts[i] if i < len(texts) else ""
            score = float(scores[i]) if i < len(scores) else 1.0
            clean = ''.join([c for c in str(text).upper().strip() if c.isalnum() or c in "-."])
            if clean:
                detections.append((np.array(poly), clean, score))
    else:
        # fallback to standard paddleocr formats (new/old)
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

    # Split multi-char detections into single-char entries (use the same split_chars routine)
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
            # split evenly across x-range
            pieces = split_chars(poly, text) 
            for box, ch, conf_e in pieces:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                xc = sum(xs) / len(xs)
                yc = sum(ys) / len(ys)
                char_entries.append((xc, yc, ch, score, box))

    if not char_entries:
        return "", [], detections

    # Compute vertical spread; if small, treat as single row
    y_coords = [e[1] for e in char_entries]
    y_min, y_max = min(y_coords), max(y_coords)
    h_crop = crop_img.shape[0] if crop_img is not None else 1
    if (y_max - y_min) < max(2.0, min_row_gap_ratio * h_crop):
        char_entries.sort(key=lambda x: x[0])
        plate = "".join([c for _, _, c, _, _ in char_entries])
        row0 = [(x, y, ch) for x, y, ch, _, _ in char_entries]
        return plate, [row0], detections

    # Otherwise attempt 1D k-means with k=2 on y_centers (simple implementation)
    ys = [e[1] for e in char_entries]
    # initialize centroids as min and max
    c1 = min(ys)
    c2 = max(ys)
    for _ in range(20):
        group1, group2 = [], []
        for idx, yv in enumerate(ys):
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

    # Assign entries into clusters
    cluster1, cluster2 = [], []
    for entry in char_entries:
        xc, yc, ch, conf, box = entry
        if abs(yc - c1) <= abs(yc - c2):
            cluster1.append(entry)
        else:
            cluster2.append(entry)

    # Decide which cluster is top (smaller y mean)
    mean1 = sum([e[1] for e in cluster1]) / len(cluster1) if cluster1 else float('inf')
    mean2 = sum([e[1] for e in cluster2]) / len(cluster2) if cluster2 else float('inf')

    if mean1 <= mean2:
        top_cluster, bottom_cluster = cluster1, cluster2
    else:
        top_cluster, bottom_cluster = cluster2, cluster1

    # Sort each cluster left->right by x_center
    top_cluster.sort(key=lambda e: e[0])
    bottom_cluster.sort(key=lambda e: e[0])

    # Build strings per row
    top_row = "".join([e[2] for e in top_cluster])
    bottom_row = "".join([e[2] for e in bottom_cluster])

    # Combine into final plate string (concatenate rows in top->bottom order)
    plate_combined = top_row + bottom_row

    # rows_list returns lists of tuples (x_center, y_center, char) for downstream use
    rows_list = [
        [(e[0], e[1], e[2]) for e in top_cluster],
        [(e[0], e[1], e[2]) for e in bottom_cluster]
    ]

    return plate_combined, rows_list, detections

# --------------------------
# Crop plate using YOLO
# --------------------------
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

# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    full_img = cv2.imread(INPUT_IMAGE)
    if full_img is None:
        raise ValueError("Cannot load input image.")

    plate_crop = crop_plate(full_img)
    cv2.imwrite("plate_crop_debug.jpg", plate_crop)

    plate_text, rows_list, raw_dets = read_plate_from_crop(plate_crop)
    rows_count = len(rows_list)

    if rows_count == 1:
        # Long car plate (single row)
        full = "".join([c for _,_,c in rows_list[0]])
        full = full.replace(".", "")  # Remove dot if exists

        # Split: everything until dash = region/series
        if "-" in full:
            left, right = full.split("-", 1)
            formatted = f"{left}-{right}"
        else:
            # no dash detected → just return cleaned
            formatted = full

        top_row = full
        bottom_row = ""
    elif rows_count >= 2:
        # Short plate (2-row)
        top_row = "".join([c for _,_,c in rows_list[0]])
        bottom_row = "".join([c for _,_,c in rows_list[1]])
        formatted = format_vn_plate(top_row, bottom_row)
    else:
        raise ValueError("OCR failed to detect characters properly")

    print("Top row:", top_row)
    print("Bottom row:", bottom_row)
    print("Detected:", plate_text)
    print("Final:", formatted)
    print("Cropped plate saved: plate_crop_debug.jpg")

    send_plate_event(formatted, confidence=1.0)
