# from paddleocr import PaddleOCR
# from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np

import onnxruntime as ort


from huggingface_hub import hf_hub_download

# Download English models
det_path = hf_hub_download("monkt/paddleocr-onnx", "detection/v5/det.onnx")
rec_path = hf_hub_download("monkt/paddleocr-onnx", "languages/english/rec.onnx")
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/english/dict.txt")

# --------------------------
# CONFIG
# --------------------------
# SHORT_MODEL_PATH = "short_plate_ds/runs/detect/train8/weights/best.pt"   # short plate YOLO model
# LONG_MODEL_PATH  = "long_plate_ds/runs/detect/train2/weights/best.pt"  # long plate YOLO model
INPUT_IMAGE = "test1.jpg"
CONF_TH = 0.40
PADDING = 8

# --------------------------
# Init models
# --------------------------

# short_model = YOLO(SHORT_MODEL_PATH)
# long_model = YOLO(LONG_MODEL_PATH)

session_short = ort.InferenceSession("short.onnx")
session_long = ort.InferenceSession("long.onnx")

input_name_short = session_short.get_inputs()[0].name
output_name_short = session_short.get_outputs()[0].name

input_name_long = session_long.get_inputs()[0].name
output_name_long = session_long.get_outputs()[0].name

# ocr = PaddleOCR(
#     lang='en',
#     # use_gpu=False,
#     # show_log=False,
#     use_angle_cls=False,
#     # det_model_dir="./paddle_models/det",
#     # rec_model_dir="./paddle_models/rec"
# )

ocr = RapidOCR(
    det_model_path=det_path,
    rec_model_path=rec_path,
    rec_keys_path=dict_path
)

class ONNXBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy      # [x1,y1,x2,y2] (pixel)
        self.conf = conf
        self.cls = cls

# def autocorrect(ch):
#     corrections = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
#     return corrections.get(ch, ch)

def preprocess(image):
    img = cv2.resize(image, (640, 640))  # Resize to model input size
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Change to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def parse_yolov8_onnx(output, orig_w, orig_h, conf_th):
    preds = output[0].squeeze()

    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes = []

    sx = orig_w / 640.0
    sy = orig_h / 640.0

    for p in preds:
        if len(p) != 5:
            continue

        cx, cy, w, h, conf = p
        if conf < conf_th:
            continue

        # YOLO output theo ảnh 640x640
        x1 = (cx - w / 2) * sx
        y1 = (cy - h / 2) * sy
        x2 = (cx + w / 2) * sx
        y2 = (cy + h / 2) * sy

        boxes.append(
            ONNXBox(
                xyxy=[x1, y1, x2, y2],
                conf=float(conf),
                cls=0
            )
        )

    return boxes

def run_session(session, input_tensor):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session.run([output_name], {input_name: input_tensor})

def detect_plate_bbox(img, try_long_first=False):
    sessions = [session_long, session_short] if try_long_first else [session_short, session_long]

    input_tensor = preprocess(img)
    img_h, img_w = img.shape[:2]

    for sess in sessions:
        results = run_session(sess, input_tensor)
        # print(results)
        boxes = parse_yolov8_onnx(results, img_w, img_h, CONF_TH)
        print(boxes)
        if boxes:
            return max(boxes, key=lambda b: b.conf)
    return None

def normalize_poly(poly):
    """
    RapidOCR polygon → np.ndarray shape (4, 2)
    """
    if poly is None:
        return None

    poly = np.asarray(poly, dtype=np.float32)

    if poly.ndim == 2 and poly.shape == (4, 2):
        return poly

    if poly.ndim == 2 and poly.shape[1] == 2 and poly.shape[0] >= 4:
        return poly[:4]

    return None

def split_chars(poly, text):
    """
    Split a multi-character OCR box into per-character boxes
    poly: (4,2)
    """
    xs = poly[:, 0]
    ys = poly[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    width = xmax - xmin
    n = len(text)

    char_boxes = []
    for i, ch in enumerate(text):
        cx1 = xmin + i * width / n
        cx2 = xmin + (i + 1) * width / n
        box = np.array([
            [cx1, ymin],
            [cx2, ymin],
            [cx2, ymax],
            [cx1, ymax],
        ], dtype=np.float32)
        char_boxes.append((box, ch, 1.0))

    return char_boxes

def read_plate_from_crop(crop_img, min_row_gap_ratio=0.08):
    """
    RapidOCR plate reader
    """
    raw = ocr(crop_img, cls=False)
    print("RapidOCR raw:", raw)

    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) < 1:
        return "", [], []

    # RapidOCR format: [detections, timing]
    detections_raw = raw[0]
    if not isinstance(detections_raw, list):
        return "", [], []

    detections = []
    for item in detections_raw:
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            continue

        poly, text, conf = item
        clean = ''.join(
            c for c in text.upper().strip()
            if c.isalnum() or c in "-."
        )
        if clean:
            detections.append((np.array(poly), clean, float(conf)))

    if not detections:
        return "", [], []

    # --- split to characters ---
    char_entries = []
    for poly, text, score in detections:
        poly = normalize_poly(poly)
        if poly is None:
            continue

        if len(text) == 1:
            xc = float(poly[:, 0].mean())
            yc = float(poly[:, 1].mean())
            char_entries.append((xc, yc, text, score, poly.tolist()))
        else:
            for box, ch, _ in split_chars(poly, text):
                box = normalize_poly(box)
                if box is None:
                    continue
                xc = float(box[:, 0].mean())
                yc = float(box[:, 1].mean())
                char_entries.append((xc, yc, ch, score, box.tolist()))

    if not char_entries:
        return "", [], detections

    # --- single row vs two rows ---
    y_vals = [e[1] for e in char_entries]
    y_min, y_max = min(y_vals), max(y_vals)
    h_crop = crop_img.shape[0]

    if (y_max - y_min) < max(2.0, min_row_gap_ratio * h_crop):
        char_entries.sort(key=lambda e: e[0])
        plate = "".join(e[2] for e in char_entries)
        rows = [[(e[0], e[1], e[2]) for e in char_entries]]
        return plate, rows, detections

    # --- 2-row k-means (1D on y) ---
    ys = [e[1] for e in char_entries]
    c1, c2 = min(ys), max(ys)

    for _ in range(20):
        g1, g2 = [], []
        for y in ys:
            (g1 if abs(y - c1) <= abs(y - c2) else g2).append(y)
        if not g1 or not g2:
            break
        nc1, nc2 = sum(g1)/len(g1), sum(g2)/len(g2)
        if abs(nc1 - c1) < 1e-3 and abs(nc2 - c2) < 1e-3:
            break
        c1, c2 = nc1, nc2

    top, bottom = [], []
    for e in char_entries:
        (top if abs(e[1] - c1) <= abs(e[1] - c2) else bottom).append(e)

    if sum(e[1] for e in top)/len(top) > sum(e[1] for e in bottom)/len(bottom):
        top, bottom = bottom, top

    top.sort(key=lambda e: e[0])
    bottom.sort(key=lambda e: e[0])

    plate = "".join(e[2] for e in top + bottom)
    rows = [
        [(e[0], e[1], e[2]) for e in top],
        [(e[0], e[1], e[2]) for e in bottom],
    ]

    return plate, rows, detections


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

    x1, y1, x2, y2 = map(int, best_box.xyxy)

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
    print("Plate text: ", plate_text)
    print("Rows list: ", rows_list)
    print("Raw detections: ", raw_dets)

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
