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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_image_index(images_dir: str):
    """
    Builds a fast lookup dictionary for all image files in images_dir.
    Maps various stem/filename variations to the full path.
    """
    image_index = {}
    if not images_dir or not os.path.isdir(images_dir):
        return image_index

    for root, _, files in os.walk(images_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, f)
                stem = os.path.splitext(f)[0]

                # Map full filename: 'card_1.jpg'
                image_index[f] = full_path
                image_index[f.lower()] = full_path

                # Map stem without ext: 'card_1'
                image_index[stem] = full_path
                image_index[stem.lower()] = full_path

                # If double extension like 'card_1.jpg':
                if any(stem.lower().endswith(e) for e in IMAGE_EXTENSIONS):
                    inner_stem = os.path.splitext(stem)[0]
                    image_index[inner_stem] = full_path
                    image_index[inner_stem.lower()] = full_path

    return image_index

def resolve_directories(data_dir: str = None, labels_dir: str = None, images_dir: str = None):
    """
    Intelligently resolves labels_dir and images_dir from user input.
    Supports:
      1. Explicit --labels_dir and --images_dir
      2. Root folder --data_dir containing 'labels' and 'images' subdirectories
      3. User passing --data_dir pointing directly to either 'labels' or 'images'
    """
    if data_dir:
        data_dir = os.path.abspath(data_dir)

        # Check if data_dir has 'labels' and 'images' subdirectories
        sub_labels = os.path.join(data_dir, "labels")
        sub_images = os.path.join(data_dir, "images")

        if os.path.isdir(sub_labels) and not labels_dir:
            labels_dir = sub_labels
        if os.path.isdir(sub_images) and not images_dir:
            images_dir = sub_images

        # Check if data_dir itself is the 'labels' folder
        base_name = os.path.basename(data_dir).lower()
        if base_name == "labels" and not labels_dir:
            labels_dir = data_dir
            parent = os.path.dirname(data_dir)
            cand_img = os.path.join(parent, "images")
            if os.path.isdir(cand_img) and not images_dir:
                images_dir = cand_img

        # Check if data_dir itself is the 'images' folder
        elif base_name == "images" and not images_dir:
            images_dir = data_dir
            parent = os.path.dirname(data_dir)
            cand_lbl = os.path.join(parent, "labels")
            if os.path.isdir(cand_lbl) and not labels_dir:
                labels_dir = cand_lbl

        # Fallback: if labels_dir is still not found, search for json files in data_dir
        if not labels_dir:
            labels_dir = data_dir
        if not images_dir:
            images_dir = data_dir

    return labels_dir, images_dir

def find_image_for_json(j_path: str, data: dict, image_index: dict, images_dir: str):
    """
    Finds matching image path using fast lookup index or JSON metadata.
    """
    j_basename = os.path.basename(j_path)
    j_stem = os.path.splitext(j_basename)[0]

    # 1. Lookup stem directly ('card_1')
    if j_stem in image_index:
        return image_index[j_stem]
    if j_stem.lower() in image_index:
        return image_index[j_stem.lower()]

    # 2. If json is named 'card_1.jpg.json' -> stem is 'card_1.jpg'
    if j_stem.endswith((".jpg", ".jpeg", ".png")):
        clean_stem = os.path.splitext(j_stem)[0]
        if clean_stem in image_index:
            return image_index[clean_stem]

    # 3. Check inside JSON fields: 'image_name', 'filename', etc.
    for k in ["image_path", "image_name", "filename", "file_name", "image"]:
        val = data.get(k)
        if isinstance(val, str) and val.strip():
            b = os.path.basename(val)
            if b in image_index:
                return image_index[b]
            b_stem = os.path.splitext(b)[0]
            if b_stem in image_index:
                return image_index[b_stem]

    # 4. Try stripping common suffixes like '_label', '_annotation'
    for suffix in ["_label", "-label", "_annot", "_metadata"]:
        if j_stem.lower().endswith(suffix):
            cand = j_stem[:-len(suffix)]
            if cand in image_index:
                return image_index[cand]
            if cand.lower() in image_index:
                return image_index[cand.lower()]

    # 5. Direct check in images_dir if index missed it
    if images_dir and os.path.isdir(images_dir):
        for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            cand_p = os.path.join(images_dir, j_stem + ext)
            if os.path.isfile(cand_p):
                return cand_p

    return None

def crop_fields_from_dataset(data_dir: str = None, labels_dir: str = None, images_dir: str = None, val_split: float = 0.15):
    """
    Prepares dataset by cropping text bounding boxes from matched card images.
    """
    os.makedirs(config.CROPS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    labels_dir, images_dir = resolve_directories(data_dir, labels_dir, images_dir)

    print(f"[*] Labels directory: {labels_dir}")
    print(f"[*] Images directory: {images_dir}")

    if not labels_dir or not os.path.isdir(labels_dir):
        print(f"[!] Labels directory does not exist: {labels_dir}")
        return

    # Index all images in images_dir
    print("[*] Indexing images...")
    image_index = build_image_index(images_dir)
    print(f"    -> Found {len(set(image_index.values()))} image files in images directory.")

    # Find all JSON files in labels_dir
    json_files = glob.glob(os.path.join(labels_dir, "**", "*.json"), recursive=True)
    print(f"    -> Found {len(json_files)} JSON annotation files.")

    if not json_files:
        print(f"[!] No JSON files found in {labels_dir}")
        return

    samples = []
    crop_counter = 0
    missing_counter = 0
    sample_images = list(set(image_index.values()))[:5]

    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Failed to parse {j_path}: {e}")
            continue

        img_path = find_image_for_json(j_path, data, image_index, images_dir)

        if not img_path or not os.path.isfile(img_path):
            missing_counter += 1
            if missing_counter <= 3:
                print(f"[-] Image not found for label: {os.path.basename(j_path)}")
                if sample_images:
                    print(f"    (Sample available images: {[os.path.basename(p) for p in sample_images]})")
            elif missing_counter == 4:
                print("[-] (Suppressing further missing image messages...)")
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

                    # Record sample
                    samples.append((crop_filepath, val))
                    crop_counter += 1

        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")

    print(f"\n[+] Extraction Summary:")
    print(f"    - Total JSON files: {len(json_files)}")
    print(f"    - Successfully processed images: {len(json_files) - missing_counter}")
    print(f"    - Missing images: {missing_counter}")
    print(f"    - Total cropped text regions: {len(samples)}")

    if not samples:
        print("[!] No text samples could be extracted.")
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

    print(f"[+] Saved {len(train_samples)} training samples to: {config.TRAIN_LABEL_FILE}")
    print(f"[+] Saved {len(val_samples)} validation samples to: {config.VAL_LABEL_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop dataset fields for lightweight OCR training")
    parser.add_argument("--data_dir", type=str, default=None, help="Root dataset folder containing 'labels' and 'images' folders")
    parser.add_argument("--labels_dir", type=str, default=None, help="Explicit path to labels folder")
    parser.add_argument("--images_dir", type=str, default=None, help="Explicit path to images folder")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set ratio (default 0.15)")
    args = parser.parse_args()

    if not args.data_dir and not args.labels_dir:
        parser.error("Please specify either --data_dir or --labels_dir")

    crop_fields_from_dataset(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        val_split=args.val_split
    )
