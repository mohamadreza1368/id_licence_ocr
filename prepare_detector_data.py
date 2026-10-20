import os
import sys
import json
import random
import shutil
from PIL import Image
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def extract_clean_stem(filename: str) -> str:
    stem = filename
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    for ext in VALID_IMG_EXTS:
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem.strip()

def build_images_lookup(images_dir: str):
    lookup = {}
    if not images_dir or not os.path.isdir(images_dir):
        return lookup

    for root, _, files in os.walk(images_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_IMG_EXTS:
                full_path = os.path.abspath(os.path.join(root, f))
                clean_stem = extract_clean_stem(f)
                lookup[clean_stem] = full_path
                lookup[clean_stem.lower()] = full_path
                lookup[f] = full_path
                lookup[f.lower()] = full_path
    return lookup

def convert_to_yolo_format(
    labels_dir: str,
    images_dir: str,
    output_dir: str = "detector_data",
    val_split: float = 0.15
):
    """
    Converts full-card JSON annotations into standard YOLO detection format.
    Generates:
      - output_dir/images/train/ & output_dir/images/val/
      - output_dir/labels/train/ & output_dir/labels/val/
      - output_dir/dataset.yaml (YOLO config file)
    """
    labels_dir = os.path.abspath(labels_dir)
    images_dir = os.path.abspath(images_dir)
    output_dir = os.path.abspath(output_dir)

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 65)
    print("  Field Detector Dataset Preparation (Full Card Object Detection)")
    print("=" * 65)
    print(f"[*] Labels Directory : {labels_dir}")
    print(f"[*] Images Directory : {images_dir}")
    print(f"[*] Output Directory : {output_dir}")

    # Build image lookup index
    images_lookup = build_images_lookup(images_dir)
    print(f"[*] Found {len(set(images_lookup.values()))} card images.")

    # Collect JSON files
    json_files = []
    for root, _, files in os.walk(labels_dir):
        for f in files:
            if f.lower().endswith(".json"):
                json_files.append(os.path.join(root, f))

    print(f"[*] Found {len(json_files)} JSON annotation files.")
    if not json_files:
        print("[!] No JSON files found.")
        return

    # Pass 1: Discover all unique field classes dynamically
    class_set = set()
    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for fld in data.get("fields", []):
                name = str(fld.get("name", "")).strip()
                if name:
                    class_set.add(name)
        except Exception:
            pass

    class_names = sorted(list(class_set))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"[*] Discovered {len(class_names)} field classes:")
    for idx, name in enumerate(class_names):
        print(f"    - Class {idx}: '{name}'")

    # Pass 2: Pair images, calculate normalized bboxes, and write YOLO labels
    records = []
    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        j_stem = extract_clean_stem(os.path.basename(j_path))
        img_path = images_lookup.get(j_stem) or images_lookup.get(j_stem.lower())
        if not img_path:
            # Check inside JSON fields
            for k in ["image_name", "filename", "image_path", "image"]:
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    img_path = images_lookup.get(extract_clean_stem(os.path.basename(v)))
                    if img_path:
                        break

        if not img_path or not os.path.isfile(img_path):
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            use_updated = True
            proc_info = data.get("processed_image", {})
            if proc_info:
                pw = proc_info.get("width", 0)
                use_updated = (abs(img_w - pw) < 50)

            boxes = []
            for fld in data.get("fields", []):
                fname = str(fld.get("name", "")).strip()
                if fname not in class_to_id:
                    continue

                bbox_data = fld.get("updated_bbox") if use_updated else fld.get("original_bbox")
                if not bbox_data or "points" not in bbox_data:
                    bbox_data = fld.get("updated_bbox") or fld.get("original_bbox")

                if not bbox_data or "points" not in bbox_data:
                    continue

                pts = bbox_data["points"]
                if len(pts) < 2:
                    continue

                xs = [int(round(p[0])) for p in pts]
                ys = [int(round(p[1])) for p in pts]

                xmin = max(0, min(xs))
                xmax = min(img_w, max(xs))
                ymin = max(0, min(ys))
                ymax = min(img_h, max(ys))

                bw = xmax - xmin
                bh = ymax - ymin
                if bw <= 2 or bh <= 2:
                    continue

                # Normalized coordinates for YOLO: <class_id> <x_center> <y_center> <width> <height>
                xc = (xmin + xmax) / 2.0 / img_w
                yc = (ymin + ymax) / 2.0 / img_h
                nw = bw / img_w
                nh = bh / img_h

                boxes.append(f"{class_to_id[fname]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if boxes:
                records.append((img_path, boxes))

        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")

    print(f"\n[+] Successfully paired {len(records)} full card images with field bounding boxes.")
    if not records:
        print("[!] No valid samples could be prepared.")
        return

    # Shuffle and split into Train / Val
    random.seed(42)
    random.shuffle(records)
    split_idx = int(len(records) * (1 - val_split)) if val_split > 0 else len(records)
    train_records = records[:split_idx]
    val_records = records[split_idx:] if val_split > 0 else records[:min(5, len(records))]

    def save_split(split_items, target_img_dir, target_lbl_dir):
        for img_src, b_list in split_items:
            base = os.path.basename(img_src)
            stem = os.path.splitext(base)[0]
            # Copy or symlink image
            dest_img = os.path.join(target_img_dir, base)
            if not os.path.exists(dest_img):
                shutil.copy2(img_src, dest_img)

            # Write YOLO label file
            dest_lbl = os.path.join(target_lbl_dir, stem + ".txt")
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                for b_str in b_list:
                    lf.write(b_str + "\n")

    print("[*] Copying images and writing YOLO annotation files...")
    save_split(train_records, train_img_dir, train_lbl_dir)
    save_split(val_records, val_img_dir, val_lbl_dir)

    # Generate dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {output_dir}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n\n")
        yf.write(f"nc: {len(class_names)}\n")
        yf.write(f"names: {class_names}\n")

    print(f"[+] Dataset preparation complete!")
    print(f"    - Training images   : {len(train_records)}")
    print(f"    - Validation images : {len(val_records)}")
    print(f"    - YOLO Config File  : {yaml_path}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Iranian National ID dataset to YOLO Field Detection format")
    parser.add_argument("--data_dir", type=str, default=None, help="Root folder containing 'labels' and 'images'")
    parser.add_argument("--labels_dir", type=str, default=None, help="Direct path to labels folder")
    parser.add_argument("--images_dir", type=str, default=None, help="Direct path to images folder")
    parser.add_argument("--output_dir", type=str, default="detector_data", help="Output directory for YOLO dataset")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set split ratio (default 0.15)")
    args = parser.parse_args()

    # Auto-resolve paths if data_dir provided
    lbl_dir = args.labels_dir
    img_dir = args.images_dir
    if args.data_dir:
        cand_l = os.path.join(args.data_dir, "labels")
        cand_i = os.path.join(args.data_dir, "images")
        if os.path.isdir(cand_l):
            lbl_dir = cand_l
        if os.path.isdir(cand_i):
            img_dir = cand_i

    if not lbl_dir or not img_dir:
        print("[!] Error: Please specify --labels_dir and --images_dir (or --data_dir)")
        sys.exit(1)

    convert_to_yolo_format(
        labels_dir=lbl_dir,
        images_dir=img_dir,
        output_dir=args.output_dir,
        val_split=args.val_split
    )
