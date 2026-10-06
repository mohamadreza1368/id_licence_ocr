import os
import sys
import json
import glob
import random
from PIL import Image
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def crop_fields_from_dataset(raw_data_dir: str, val_split: float = 0.15):
    """
    Scans raw_data_dir for pairs of .json and corresponding image files.
    Crops each field in 'fields' list, saves crops to data/crops/,
    and writes train.txt and val.txt.
    """
    os.makedirs(config.CROPS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    json_files = glob.glob(os.path.join(raw_data_dir, "**", "*.json"), recursive=True)
    if not json_files:
        print(f"[!] No JSON files found in {raw_data_dir}")
        return

    print(f"[*] Found {len(json_files)} JSON annotation files in {raw_data_dir}")

    samples = []
    crop_counter = 0

    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Failed to parse {j_path}: {e}")
            continue

        # Find matching image file
        base_no_ext = os.path.splitext(j_path)[0]
        img_path = None
        for ext in image_extensions:
            candidate = base_no_ext + ext
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            # Also check if image is specified in json or directory
            parent_dir = os.path.dirname(j_path)
            cand_files = os.listdir(parent_dir)
            for f_name in cand_files:
                if any(f_name.lower().endswith(ext) for ext in image_extensions):
                    # Check same prefix
                    if os.path.splitext(f_name)[0] == os.path.splitext(os.path.basename(j_path))[0]:
                        img_path = os.path.join(parent_dir, f_name)
                        break

        if not img_path or not os.path.exists(img_path):
            print(f"[-] Image not found for {j_path}, skipping.")
            continue

        try:
            with Image.open(img_path) as full_img:
                img_w, img_h = full_img.size

                # Determine which bbox to use (original vs updated)
                processed_info = data.get("processed_image", {})
                proc_w = processed_info.get("width", 0)

                # If current image resolution matches processed_image, prefer updated_bbox
                use_updated = (abs(img_w - proc_w) < 50) if proc_w else True

                fields = data.get("fields", [])
                for idx, field in enumerate(fields):
                    val = str(field.get("value", "")).strip()
                    if not val:
                        continue

                    # Select bbox
                    bbox_data = None
                    if use_updated and "updated_bbox" in field:
                        bbox_data = field["updated_bbox"]
                    elif "original_bbox" in field:
                        bbox_data = field["original_bbox"]
                    elif "updated_bbox" in field:
                        bbox_data = field["updated_bbox"]

                    if not bbox_data or "points" not in bbox_data:
                        continue

                    points = bbox_data["points"]
                    if len(points) < 2:
                        continue

                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]

                    xmin = min(xs)
                    xmax = max(xs)
                    ymin = min(ys)
                    ymax = max(ys)

                    # Add slight padding around text box
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    pad_x = int(box_w * 0.02)
                    pad_y = int(box_h * 0.05)

                    xmin = max(0, xmin - pad_x)
                    ymin = max(0, ymin - pad_y)
                    xmax = min(img_w, xmax + pad_x)
                    ymax = min(img_h, ymax + pad_y)

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    crop_img = full_img.crop((xmin, ymin, xmax, ymax))

                    # Save crop
                    crop_filename = f"crop_{crop_counter:06d}_{field.get('name', 'field')}.png"
                    crop_filepath = os.path.join(config.CROPS_DIR, crop_filename)
                    crop_img.save(crop_filepath)

                    # Record sample: relative or absolute path and label
                    samples.append((crop_filepath, val))
                    crop_counter += 1

        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")

    print(f"[+] Total cropped text regions created: {len(samples)}")

    if not samples:
        print("[!] No samples could be extracted.")
        return

    # Shuffle and split into train / val
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * (1 - val_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Write train.txt
    with open(config.TRAIN_LABEL_FILE, "w", encoding="utf-8") as f:
        for p, label in train_samples:
            f.write(f"{p}\t{label}\n")

    # Write val.txt
    with open(config.VAL_LABEL_FILE, "w", encoding="utf-8") as f:
        for p, label in val_samples:
            f.write(f"{p}\t{label}\n")

    print(f"[+] Written {len(train_samples)} training samples to: {config.TRAIN_LABEL_FILE}")
    print(f"[+] Written {len(val_samples)} validation samples to: {config.VAL_LABEL_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop dataset fields for lightweight OCR training")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing json and card images")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set ratio (default 0.15)")
    args = parser.parse_args()

    crop_fields_from_dataset(args.data_dir, args.val_split)
