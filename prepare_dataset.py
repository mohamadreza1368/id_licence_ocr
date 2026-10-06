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

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"]

def find_matching_image(j_path: str, data: dict, raw_data_dir: str):
    """
    Intelligently finds the matching image for a given json file across
    multiple naming conventions and folder structures.
    """
    j_dir = os.path.dirname(os.path.abspath(j_path))
    j_basename = os.path.basename(j_path)
    j_stem = os.path.splitext(j_basename)[0]

    # Strategy 1: Double extension (e.g. 'sample.jpg.json' -> 'sample.jpg')
    base_no_ext = os.path.splitext(j_path)[0]
    if os.path.isfile(base_no_ext):
        return base_no_ext

    # Strategy 2: Exact stem in the same folder (e.g. 'sample.json' -> 'sample.jpg')
    for ext in IMAGE_EXTENSIONS:
        cand = os.path.join(j_dir, j_stem + ext)
        if os.path.isfile(cand):
            return cand

    # Strategy 3: Check inside JSON metadata for image name / path
    keys_to_check = ["image_path", "image_name", "filename", "file_name", "image", "img", "image_file"]
    for key in keys_to_check:
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            # Check direct path
            if os.path.isfile(val):
                return val
            # Check relative to j_dir
            cand1 = os.path.join(j_dir, val)
            if os.path.isfile(cand1):
                return cand1
            # Check relative to raw_data_dir
            cand2 = os.path.join(raw_data_dir, val)
            if os.path.isfile(cand2):
                return cand2
            # Check basename in j_dir
            cand3 = os.path.join(j_dir, os.path.basename(val))
            if os.path.isfile(cand3):
                return cand3

    # Check nested dictionary metadata (e.g. processed_image, original_image)
    for nested_key in ["processed_image", "original_image", "image_info", "metadata"]:
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            for subkey in ["path", "filename", "name", "file", "image_path", "image_name"]:
                val = nested.get(subkey)
                if isinstance(val, str) and val.strip():
                    for check_dir in [j_dir, raw_data_dir]:
                        c = os.path.join(check_dir, os.path.basename(val))
                        if os.path.isfile(c):
                            return c

    # Strategy 4: Sibling directories (e.g. json in 'labels/' or 'annotations/', images in 'images/')
    parent_dir = os.path.dirname(j_dir)
    possible_img_dirs = [
        j_dir,
        parent_dir,
        os.path.join(parent_dir, "images"),
        os.path.join(parent_dir, "image"),
        os.path.join(parent_dir, "imgs"),
        os.path.join(parent_dir, "img"),
        os.path.join(parent_dir, "processed"),
        os.path.join(parent_dir, "original"),
        os.path.join(parent_dir, "cards"),
        os.path.join(raw_data_dir, "images"),
        os.path.join(raw_data_dir, "imgs"),
    ]
    for p_dir in possible_img_dirs:
        if os.path.isdir(p_dir):
            for ext in IMAGE_EXTENSIONS:
                cand = os.path.join(p_dir, j_stem + ext)
                if os.path.isfile(cand):
                    return cand

    # Strategy 5: Suffix variations (e.g. 'sample_label.json' -> 'sample.jpg', or 'sample.json' -> 'sample_processed.jpg')
    cleaned_stem = j_stem
    for suffix in ["_label", "_annot", "_annotation", "_metadata", "-label", "_json"]:
        if cleaned_stem.lower().endswith(suffix):
            cleaned_stem = cleaned_stem[:-len(suffix)]
            break

    for p_dir in [j_dir, parent_dir]:
        if os.path.isdir(p_dir):
            for ext in IMAGE_EXTENSIONS:
                # e.g. cleaned stem
                cand = os.path.join(p_dir, cleaned_stem + ext)
                if os.path.isfile(cand):
                    return cand
                # e.g. _processed or _original suffixes
                for variant in [f"{cleaned_stem}_processed{ext}", f"processed_{cleaned_stem}{ext}", f"{cleaned_stem}_original{ext}"]:
                    cand_v = os.path.join(p_dir, variant)
                    if os.path.isfile(cand_v):
                        return cand_v

    # Strategy 6: If the directory contains exactly one image, pair them
    if os.path.isdir(j_dir):
        files_in_dir = os.listdir(j_dir)
        imgs_in_dir = [f for f in files_in_dir if any(f.lower().endswith(e.lower()) for e in IMAGE_EXTENSIONS)]
        if len(imgs_in_dir) == 1:
            return os.path.join(j_dir, imgs_in_dir[0])

    # Strategy 7: Recursive search in raw_data_dir for file with matching stem
    for ext in IMAGE_EXTENSIONS:
        matched = glob.glob(os.path.join(raw_data_dir, "**", f"{j_stem}{ext}"), recursive=True)
        if matched:
            return matched[0]
        if cleaned_stem != j_stem:
            matched2 = glob.glob(os.path.join(raw_data_dir, "**", f"{cleaned_stem}{ext}"), recursive=True)
            if matched2:
                return matched2[0]

    return None


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
    missing_counter = 0

    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Failed to parse {j_path}: {e}")
            continue

        # Intelligent image matching
        img_path = find_matching_image(j_path, data, raw_data_dir)

        if not img_path:
            missing_counter += 1
            if missing_counter <= 5:
                parent_dir = os.path.dirname(j_path)
                dir_contents = os.listdir(parent_dir)[:10] if os.path.isdir(parent_dir) else []
                print(f"[-] Image not found for: {j_path}")
                print(f"    Directory contents sample: {dir_contents}")
            elif missing_counter == 6:
                print("[-] (Suppressing further missing image logs...)")
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

    print(f"\n[+] Processing finished:")
    print(f"    - Total JSON files: {len(json_files)}")
    print(f"    - Missing images: {missing_counter}")
    print(f"    - Total cropped text regions created: {len(samples)}")

    if not samples:
        print("[!] No samples could be extracted. Please check the image path and naming.")
        return

    # Shuffle and split into train / val
    random.seed(42)
    random.shuffle(samples)

    if val_split > 0 and len(samples) > 1:
        split_idx = int(len(samples) * (1 - val_split))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
    else:
        train_samples = samples
        val_samples = []

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
