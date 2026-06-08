from pathlib import Path
import json
from datetime import datetime

import cv2
from ultralytics import YOLO


# ===== CONFIG =====
DEFAULT_MODEL = "models/yolo26/yolo26n.onnx"
IMG_SIZE = 640


def predict_image(image_path: str, model_path: str = DEFAULT_MODEL):
    """
    Predict ảnh, lưu artifacts và trả về kết quả.
    """
    
    model = YOLO(model_path, task='detect')
    image_path = Path(image_path)

    # Đọc ảnh gốc
    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Predict
    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        verbose=False
    )[0]

    # Ảnh đã detect
    detected_img = results.plot(
        line_width=1,
        font_size=0.5
    )

    detected_img = cv2.cvtColor(
        detected_img,
        cv2.COLOR_BGR2RGB
    )

    # ==========================
    # SAVE ARTIFACTS
    # ==========================

    output_dir = Path("reports/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = image_path.stem
    model_name = Path(model_path).stem

    pred_image_path = output_dir / f"{image_stem}_{model_name}_pred.png"
    pred_json_path = output_dir / f"{image_stem}_{model_name}_result.json"


    # Save image
    cv2.imwrite(
        str(pred_image_path),
        cv2.cvtColor(
            detected_img,
            cv2.COLOR_BGR2RGB
        )
    )

    # Build JSON
    detections = []

    for box in results.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        class_name = results.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 6),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            }
        )

    json_result = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_path.name,
        "model_name": Path(model_path).name,
        "num_detections": len(detections),
        "detections": detections
    }

    # Save JSON
    with open(
        pred_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_result,
            f,
            indent=4,
            ensure_ascii=False
        )

    return {
        "image_rgb": img_rgb,
        "detected_img": detected_img,
        "result_json": json_result,
        "pred_image_path": str(pred_image_path),
        "pred_json_path": str(pred_json_path)
    }