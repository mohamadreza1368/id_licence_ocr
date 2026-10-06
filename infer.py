import os
import sys
import json
import torch
from PIL import Image
import torchvision.transforms.functional as TF

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import config
from tokenizer import CTCLabelConverter
from model import MobileOCRNet

def validate_iranian_national_id(code: str) -> bool:
    """
    Standard Iranian National ID (کد ملی) checksum validation algorithm.
    """
    # Normalize digits
    persian_to_eng = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    code = code.translate(persian_to_eng).strip()

    if not code.isdigit() or len(code) != 10:
        return False
    # Check invalid repetition like 0000000000, 1111111111, etc.
    if len(set(code)) == 1:
        return False

    check = int(code[9])
    s = sum(int(code[i]) * (10 - i) for i in range(9))
    r = s % 11
    return (r < 2 and check == r) or (r >= 2 and check == (11 - r))

class NationalCardOCR:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.converter = CTCLabelConverter(config.ALL_CHARS)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, "best_mobile_ocr.pth")

        self.model = MobileOCRNet(num_classes=self.converter.num_classes)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[*] Loaded OCR model from {checkpoint_path}")
        else:
            print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with uninitialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, img: Image.Image):
        w, h = img.size
        scale = config.IMG_HEIGHT / float(h)
        new_w = max(config.MIN_WIDTH, min(config.MAX_WIDTH, int(w * scale)))
        if new_w % 4 != 0:
            new_w = (new_w // 4 + 1) * 4

        resized = img.resize((new_w, config.IMG_HEIGHT), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return tensor.unsqueeze(0).to(self.device)  # [1, 3, 32, W]

    def read_crop(self, img_crop: Image.Image) -> str:
        tensor = self.preprocess(img_crop.convert("RGB"))
        with torch.no_grad():
            preds = self.model.predict(tensor).cpu().numpy()
            decoded = self.converter.decode_greedy(preds)
            return decoded[0] if decoded else ""

    def extract_from_card(self, image_path: str, json_path: str = None, bbox_list: list = None):
        """
        Extracts information from a card image.
        Uses json_path or bbox_list to crop and read each field.
        """
        with Image.open(image_path) as full_img:
            img_w, img_h = full_img.size
            results = {}

            fields_to_process = []
            if json_path and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    fields_to_process = meta.get("fields", [])

            for f in fields_to_process:
                name = f.get("name", "Unknown")
                # Choose updated or original bbox
                bbox = f.get("updated_bbox") or f.get("original_bbox")
                if not bbox or "points" not in bbox:
                    continue

                pts = bbox["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                xmin = max(0, min(xs))
                ymin = max(0, min(ys))
                xmax = min(img_w, max(xs))
                ymax = min(img_h, max(ys))

                crop = full_img.crop((xmin, ymin, xmax, ymax))
                text = self.read_crop(crop)

                # Validation & post-processing
                is_valid = None
                if "national" in name.lower() or "id" in name.lower():
                    is_valid = validate_iranian_national_id(text)

                results[name] = {
                    "extracted_text": text,
                    "ground_truth": f.get("value", ""),
                    "is_valid": is_valid
                }

            return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to single crop image or full card image")
    parser.add_argument("--json", type=str, default=None, help="Path to json file if passing full card")
    args = parser.parse_args()

    ocr = NationalCardOCR()
    if args.image:
        if args.json:
            res = ocr.extract_from_card(args.image, args.json)
            print("\n--- Card Extraction Results ---")
            for k, v in res.items():
                print(f"{k}: {v['extracted_text']} (GT: {v['ground_truth']})")
        else:
            with Image.open(args.image) as img:
                txt = ocr.read_crop(img)
                print(f"Recognized Text: {txt}")
    else:
        print("Please provide --image [path_to_image]")
