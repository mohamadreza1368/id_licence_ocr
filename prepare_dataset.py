import os
import sys
import json
import random
from PIL import Image
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_clean_stem(filename: str) -> str:
    """
    Extracts the clean stem name from a file without any extensions.
    Examples:
      - 'card_001.json' -> 'card_001'
      - 'card_001.jpg.json' -> 'card_001'
      - 'card_001.JPG' -> 'card_001'
      - 'card_001.jpg' -> 'card_001'
    """
    stem = filename
    # Strip .json
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    # Strip any image extension
    for ext in VALID_IMG_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem.strip()

def find_subfolder(parent: str, candidates: list):
    """
    Finds a subfolder inside parent matching any candidate name (case-insensitive).
    """
    if not parent or not os.path.isdir(parent):
        return None
    try:
        for entry in os.listdir(parent):
            full_path = os.path.join(parent, entry)
            if os.path.isdir(full_path):
                if entry.lower() in [c.lower() for c in candidates]:
                    return full_path
    except Exception:
        pass
    return None

def resolve_dataset_paths(data_dir: str = None, labels_dir: str = None, images_dir: str = None):
    """
    Resolves the exact paths for labels_dir and images_dir.
    Handles:
      - Root folder containing 'labels' and 'images' subfolders
      - Path directly pointing to 'labels' folder (finds sibling 'images')
      - Path directly pointing to 'images' folder (finds sibling 'labels')
      - Explicit --labels_dir and --images_dir arguments
    """
    label_folder_names = ["labels", "label", "annotations", "annotation", "json", "jsons"]
    image_folder_names = ["images", "image", "imgs", "img", "photos", "pics", "cards"]

    # If explicit paths provided
    if labels_dir:
        labels_dir = os.path.abspath(labels_dir)
    if images_dir:
        images_dir = os.path.abspath(images_dir)

    # If data_dir is provided
    if data_dir:
        data_dir = os.path.abspath(data_dir)

        # Case 1: data_dir has 'labels' and 'images' subfolders
        if not labels_dir:
            labels_dir = find_subfolder(data_dir, label_folder_names)
        if not images_dir:
            images_dir = find_subfolder(data_dir, image_folder_names)

        # Case 2: data_dir is the 'labels' folder itself
        base_name = os.path.basename(data_dir).lower()
        if base_name in label_folder_names and not labels_dir:
            labels_dir = data_dir
        if base_name in label_folder_names and not images_dir:
            parent_dir = os.path.dirname(data_dir)
            images_dir = find_subfolder(parent_dir, image_folder_names)

        # Case 3: data_dir is the 'images' folder itself
        if base_name in image_folder_names and not images_dir:
            images_dir = data_dir
        if base_name in image_folder_names and not labels_dir:
            parent_dir = os.path.dirname(data_dir)
            labels_dir = find_subfolder(parent_dir, label_folder_names)

        # Fallback if still not found
        if not labels_dir:
            labels_dir = data_dir
        if not images_dir:
            images_dir = data_dir

    return labels_dir, images_dir

def build_images_lookup(images_dir: str):
    """
    Indexes all image files inside images_dir (recursively) for fast O(1) matching.
    Key is the clean stem (lowercase and original), value is the absolute path.
    """
    lookup = {}
    raw_files = []

    if not images_dir or not os.path.isdir(images_dir):
        return lookup, raw_files

    for root, _, files in os.walk(images_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_IMG_EXTS:
                full_path = os.path.abspath(os.path.join(root, f))
                raw_files.append(full_path)

                clean_stem = extract_clean_stem(f)
                # Map various normalized versions of the stem
                lookup[clean_stem] = full_path
                lookup[clean_stem.lower()] = full_path
                lookup[f] = full_path
                lookup[f.lower()] = full_path

    return lookup, raw_files

def get_all_json_files(labels_dir: str):
    """
    Finds all .json files inside labels_dir using os.walk (robust against brackets/spaces/unicode).
    """
    json_list = []
    if not labels_dir or not os.path.isdir(labels_dir):
        return json_list

    for root, _, files in os.walk(labels_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_list.append(os.path.abspath(os.path.join(root, f)))

    return json_list

def find_matched_image(j_path: str, data: dict, images_lookup: dict, images_dir: str):
    """
    Locates the matching image for the json file using clean stem logic:
    1. Extract name without format extension (e.g. '123' from '123.json')
    2. Look for that exact name with .jpg inside images
    3. Check lookup dictionary
    4. Fallback to image name specified inside JSON metadata
    """
    j_filename = os.path.basename(j_path)
    stem = extract_clean_stem(j_filename)

    # 1. Direct check in images_dir for stem.jpg, stem.JPG, stem.jpeg, stem.png
    if images_dir and os.path.isdir(images_dir):
        for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
            candidate = os.path.join(images_dir, stem + ext)
            if os.path.isfile(candidate):
                return candidate

    # 2. Check the fast O(1) lookup dictionary (covers subfolders too)
    if stem in images_lookup:
        return images_lookup[stem]
    if stem.lower() in images_lookup:
        return images_lookup[stem.lower()]

    # 3. Check inside JSON metadata fields (if image filename is recorded inside json)
    for field in ["image_name", "filename", "image_path", "image", "file_name"]:
        val = data.get(field)
        if isinstance(val, str) and val.strip():
            base_val = os.path.basename(val)
            val_stem = extract_clean_stem(base_val)
            if val_stem in images_lookup:
                return images_lookup[val_stem]
            if val_stem.lower() in images_lookup:
                return images_lookup[val_stem.lower()]
            if images_dir:
                cand = os.path.join(images_dir, base_val)
                if os.path.isfile(cand):
                    return cand

    # 4. Check nested image objects in json (processed_image, original_image)
    for parent_key in ["processed_image", "original_image"]:
        parent_obj = data.get(parent_key)
        if isinstance(parent_obj, dict):
            for sub_k in ["name", "filename", "path", "image_name"]:
                v = parent_obj.get(sub_k)
                if isinstance(v, str) and v.strip():
                    v_stem = extract_clean_stem(os.path.basename(v))
                    if v_stem in images_lookup:
                        return images_lookup[v_stem]

    return None

def crop_fields_from_dataset(data_dir: str = None, labels_dir: str = None, images_dir: str = None, val_split: float = 0.15):
    os.makedirs(config.CROPS_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    labels_dir, images_dir = resolve_dataset_paths(data_dir, labels_dir, images_dir)

    print("=" * 65)
    print("  Lightweight OCR Dataset Extractor (Persian National ID Cards)")
    print("=" * 65)
    print(f"[*] Labels Directory : {labels_dir}")
    print(f"[*] Images Directory : {images_dir}")

    if not labels_dir or not os.path.isdir(labels_dir):
        print(f"\n[!] Error: Labels directory not found: {labels_dir}")
        print("    Please check the path to your dataset.")
        return

    if not images_dir or not os.path.isdir(images_dir):
        print(f"\n[!] Error: Images directory not found: {images_dir}")
        print("    Please check the path to your dataset.")
        return

    # Index images
    print("\n[*] Scanning and indexing images in images directory...")
    images_lookup, all_images = build_images_lookup(images_dir)
    print(f"    -> Found {len(all_images)} image files (.jpg / .jpeg / .png).")

    if not all_images:
        print(f"[!] No valid image files (.jpg, .png) found in: {images_dir}")
        return

    # Scan json files
    print("[*] Scanning labels directory for JSON files...")
    json_files = get_all_json_files(labels_dir)
    print(f"    -> Found {len(json_files)} JSON label files.")

    if not json_files:
        print(f"[!] No JSON files found in: {labels_dir}")
        return

    print("\n[*] Matching JSON labels with images and cropping text fields...")
    print("-" * 65)

    samples = []
    crop_counter = 0
    missing_count = 0
    matched_count = 0

    sample_image_names = [os.path.basename(p) for p in all_images[:5]]

    for idx, j_path in enumerate(json_files, 1):
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Error reading JSON {os.path.basename(j_path)}: {e}")
            continue

        # Match image
        img_path = find_matched_image(j_path, data, images_lookup, images_dir)

        if not img_path:
            missing_count += 1
            if missing_count <= 3:
                clean_name = extract_clean_stem(os.path.basename(j_path))
                print(f"[!] Could NOT match image for: {os.path.basename(j_path)}")
                print(f"    -> Extracted name without format: '{clean_name}'")
                print(f"    -> Looked for: '{clean_name}.jpg' in {images_dir}")
                print(f"    -> Sample actual image names in folder: {sample_image_names}")
            elif missing_count == 4:
                print("[!] (Suppressing further missing image notifications...)")
            continue

        matched_count += 1
        if matched_count <= 2:
            print(f"[OK] Matched: '{os.path.basename(j_path)}' -> '{os.path.basename(img_path)}'")

        try:
            with Image.open(img_path) as full_img:
                img_w, img_h = full_img.size

                # Choose bbox matching image resolution
                processed_info = data.get("processed_image", {})
                proc_w = processed_info.get("width", 0)
                use_updated = (abs(img_w - proc_w) < 50) if proc_w else True

                fields = data.get("fields", [])
                for f_idx, field in enumerate(fields):
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

                    # Round coordinate points safely
                    xs = [int(round(p[0])) for p in points]
                    ys = [int(round(p[1])) for p in points]

                    xmin = min(xs)
                    xmax = max(xs)
                    ymin = min(ys)
                    ymax = max(ys)

                    # Add slight padding
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    pad_x = int(box_w * 0.02)
                    pad_y = int(box_h * 0.05)

                    xmin = max(0, min(img_w - 1, xmin - pad_x))
                    xmax = max(xmin + 1, min(img_w, xmax + pad_x))
                    ymin = max(0, min(img_h - 1, ymin - pad_y))
                    ymax = max(ymin + 1, min(img_h, ymax + pad_y))

                    if xmax <= xmin or ymax <= ymin:
                        continue

                    crop_img = full_img.crop((xmin, ymin, xmax, ymax))

                    # Save crop
                    field_name = str(field.get("name", "field")).replace(" ", "_")
                    crop_filename = f"crop_{crop_counter:06d}_{field_name}.png"
                    crop_filepath = os.path.join(config.CROPS_DIR, crop_filename)
                    crop_img.save(crop_filepath)

                    samples.append((crop_filepath, val))
                    crop_counter += 1

        except Exception as e:
            print(f"[!] Error processing image {img_path}: {e}")

    print("-" * 65)
    print("\n[+] Processing Summary:")
    print(f"    - Total JSON labels    : {len(json_files)}")
    print(f"    - Successfully matched : {matched_count}")
    print(f"    - Missing images       : {missing_count}")
    print(f"    - Total text crops     : {len(samples)}")

    if not samples:
        print("\n[!] No text crops were generated. Please inspect the logs above.")
        return

    # Shuffle and split train / val
    random.seed(42)
    random.shuffle(samples)

    if val_split > 0 and len(samples) > 1:
        split_idx = int(len(samples) * (1 - val_split))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
    else:
        train_samples = samples
        val_samples = []

    with open(config.TRAIN_LABEL_FILE, "w", encoding="utf-8") as f:
        for p, label in train_samples:
            f.write(f"{p}\t{label}\n")

    with open(config.VAL_LABEL_FILE, "w", encoding="utf-8") as f:
        for p, label in val_samples:
            f.write(f"{p}\t{label}\n")

    print(f"\n[+] Saved {len(train_samples)} training samples to   : {config.TRAIN_LABEL_FILE}")
    print(f"[+] Saved {len(val_samples)} validation samples to : {config.VAL_LABEL_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop dataset fields for lightweight OCR training")
    parser.add_argument("--data_dir", type=str, default=None, help="Root folder containing 'labels' and 'images' subdirectories")
    parser.add_argument("--labels_dir", type=str, default=None, help="Explicit path to labels folder")
    parser.add_argument("--images_dir", type=str, default=None, help="Explicit path to images folder")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set ratio (default 0.15)")
    args = parser.parse_args()

    if not args.data_dir and not args.labels_dir:
        parser.error("Please specify --data_dir (or --labels_dir and --images_dir)")

    crop_fields_from_dataset(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        val_split=args.val_split
    )
