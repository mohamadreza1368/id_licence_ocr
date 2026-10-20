import os
import sys
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
import config
from infer import NationalCardOCR, validate_iranian_national_id

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

class FullCardOCRScanner:
    """
    End-to-End Mobile ID Card Scanner:
    Takes a raw mobile phone photo of an Iranian National ID card,
    detects the field bounding boxes (Stage 1),
    and recognizes the text with lightweight MobileOCRNet (Stage 2).
    """
    def __init__(
        self,
        detector_weights: str = None,
        recognizer_weights: str = None,
        conf_threshold: float = 0.40
    ):
        if detector_weights is None:
            detector_weights = os.path.join(config.CHECKPOINTS_DIR, "best_field_detector.pt")
        if recognizer_weights is None:
            recognizer_weights = os.path.join(config.CHECKPOINTS_DIR, "best_mobile_ocr.pth")

        self.conf_threshold = conf_threshold

        # 1. Load Text Recognizer (Stage 2)
        print("[*] Initializing Text Recognizer (MobileOCRNet)...")
        self.ocr_engine = NationalCardOCR(checkpoint_path=recognizer_weights)

        # 2. Load Field Detector (Stage 1)
        self.detector = None
        if os.path.exists(detector_weights):
            try:
                from ultralytics import YOLO
                print(f"[*] Loading Field Detector from: {detector_weights}")
                self.detector = YOLO(detector_weights)
            except ImportError:
                print("[!] Warning: 'ultralytics' not installed. Install it with: pip install ultralytics")
            except Exception as e:
                print(f"[!] Warning: Could not load detector: {e}")
        else:
            print(f"[!] Note: Field detector weights not found at: {detector_weights}")
            print("    To detect fields automatically from full mobile photos, run: python train_detector.py")

    def detect_fields(self, image_path: str):
        """
        Runs Field Detector on raw mobile photo to get bounding boxes of each field.
        Returns list of dicts: [{'name': '...', 'box': [xmin, ymin, xmax, ymax], 'conf': 0.95}, ...]
        """
        if self.detector is None:
            return []

        results = self.detector.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )
        if not results:
            return []

        res = results[0]
        detected = []
        names_dict = res.names  # {id: 'class_name'}

        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            class_name = names_dict.get(cls_id, f"field_{cls_id}")

            detected.append({
                "name": class_name,
                "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "conf": round(conf, 3)
            })

        return detected

    def scan_card(self, image_path: str, save_annotated_path: str = None):
        """
        Processes a full mobile camera photo and extracts all fields.
        """
        if not os.path.isfile(image_path):
            print(f"[!] Error: Image not found: {image_path}")
            return {}

        with Image.open(image_path) as full_img:
            full_img = full_img.convert("RGB")
            img_w, img_h = full_img.size

            # Stage 1: Detect fields
            detected_fields = self.detect_fields(image_path)

            if not detected_fields:
                print("[!] No fields detected. If detector is not yet trained, please train it with train_detector.py")
                return {}

            results = {}
            annotated_img = full_img.copy() if save_annotated_path else None
            draw = ImageDraw.Draw(annotated_img) if annotated_img else None

            # Stage 2: Crop and Recognize each detected field
            for item in detected_fields:
                fname = item["name"]
                xmin, ymin, xmax, ymax = item["box"]
                conf = item["conf"]

                # Add small adaptive padding (5%)
                bw = xmax - xmin
                bh = ymax - ymin
                pad_x = int(bw * 0.03)
                pad_y = int(bh * 0.05)

                crop_xmin = max(0, xmin - pad_x)
                crop_ymin = max(0, ymin - pad_y)
                crop_xmax = min(img_w, xmax + pad_x)
                crop_ymax = min(img_h, ymax + pad_y)

                crop = full_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
                recognized_text = self.ocr_engine.read_crop(crop)

                # Validation
                is_valid = None
                if "national" in fname.lower() or "id" in fname.lower():
                    is_valid = validate_iranian_national_id(recognized_text)

                results[fname] = {
                    "text": recognized_text,
                    "detector_confidence": conf,
                    "is_valid": is_valid,
                    "bbox": [xmin, ymin, xmax, ymax]
                }

                # Draw bounding box on visualization image
                if draw:
                    draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 255, 0), width=4)
                    draw.text((xmin, max(0, ymin - 25)), f"{fname}: {recognized_text}", fill=(255, 0, 0))

            if save_annotated_path and annotated_img:
                annotated_img.save(save_annotated_path)
                print(f"[+] Annotated visual image saved to: {save_annotated_path}")

            return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Iranian ID Card Scanner from Raw Mobile Photos")
    parser.add_argument("--image", type=str, required=True, help="Path to raw mobile photo of ID card")
    parser.add_argument("--detector", type=str, default=None, help="Path to field detector weights (.pt)")
    parser.add_argument("--recognizer", type=str, default=None, help="Path to text recognizer weights (.pth)")
    parser.add_argument("--conf", type=float, default=0.40, help="Detector confidence threshold (default: 0.40)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save annotated visualization image")
    args = parser.parse_args()

    scanner = FullCardOCRScanner(
        detector_weights=args.detector,
        recognizer_weights=args.recognizer,
        conf_threshold=args.conf
    )

    print(f"\n[*] Processing mobile card photo: {args.image}")
    results = scanner.scan_card(args.image, save_annotated_path=args.output)

    print("\n" + "=" * 55)
    print("  Extracted National ID Card Information")
    print("=" * 55)
    for field_name, info in results.items():
        val_status = ""
        if info["is_valid"] is True:
            val_status = " [VALID CHECKSUM]"
        elif info["is_valid"] is False:
            val_status = " [INVALID CHECKSUM]"

        print(f"  {field_name:<16}: {info['text']}{val_status} (conf: {info['detector_confidence']})")
    print("=" * 55 + "\n")
