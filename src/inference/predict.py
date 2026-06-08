from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from predictor import predict_image


# ===== CONFIG =====
DEFAULT_MODEL = "yolo26n"
DEFAULT_IMAGE = "data/samples/maksssksksss12.png"
SAMPLES_DIR = Path("data/samples")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Face Mask Detection Inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Image filename or path"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name (e.g.: yolo26n, yolo11s)"
    )

    return parser.parse_args()


def resolve_image_path(image_arg):

    input_path = Path(image_arg)

    if not input_path.exists():
        input_path = SAMPLES_DIR / image_arg

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {input_path}")

    return str(input_path)


def resolve_model_path(model_name):
    matches = list(
        Path("models").rglob(f"{model_name}.onnx")
    )

    if len(matches) == 0:
        raise FileNotFoundError(f"Không tìm thấy model: {model_name}")

    if len(matches) > 1:
        raise RuntimeError(f"Có nhiều model trùng tên: {model_name}")

    return str(matches[0])


def main():

    args = parse_arguments()

    image_path = resolve_image_path(args.image)
    model_path = resolve_model_path(args.model)

    print(f"\nPredicting image: {image_path}")

    result = predict_image(image_path, model_path)

    print(f"\nSaved image: {result['pred_image_path']}")
    print(f"Saved json: {result['pred_json_path']}")


    # In thông tin detection ra terminal
    print("\n=== Detection Results ===")

    detections = result["result_json"]["detections"]

    if len(detections) == 0:
        print("Không phát hiện đối tượng nào.")
    else:
        for i, det in enumerate(detections,start=1):
            print(
                f"Object {i}: "
                f"{det['class_name']} "
                f"(confidence="
                f"{det['confidence']:.4f})"
            )


    # Hiển thị ảnh
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].imshow(result["image_rgb"])
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(result["detected_img"])
    axs[1].set_title("YOLO26 Detection")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()