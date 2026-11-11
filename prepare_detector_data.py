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

from prepare_dataset import (
    VALID_IMG_EXTS,
    extract_clean_stem,
    extract_field_points,
    get_coordinate_reference,
    build_images_lookup,
    get_all_json_files,
    find_matched_image,
    resolve_dataset_paths,
)

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

    # Build image lookup index using exact stem-first logic
    print("\n[*] Scanning and indexing card images in images directory...")
    images_lookup, all_images = build_images_lookup(images_dir)
    print(f"    -> Found {len(all_images)} card images (.jpg / .jpeg / .png).")

    if not all_images:
        print(f"[!] Error: No valid image files (.jpg, .png) found in: {images_dir}")
        return

    # Collect JSON files
    print("[*] Scanning labels directory for JSON files...")
    json_files = get_all_json_files(labels_dir)
    print(f"    -> Found {len(json_files)} JSON annotation files.")

    if not json_files:
        print(f"[!] Error: No JSON files found in: {labels_dir}")
        return

    # Pass 1: Discover all unique field classes dynamically
    print("\n[*] Pass 1: Discovering field classes...")
    class_set = set()
    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for fld in data.get("fields", []):
                name = str(fld.get("name", "")).strip()
                pts = extract_field_points(fld)
                # Only include fields with valid bounding box coordinates (skips metadata like image_path with empty bbox)
                if name and pts and len(pts) >= 2:
                    class_set.add(name)
        except Exception:
            pass

    class_names = sorted(list(class_set))
    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"[+] Discovered {len(class_names)} valid detection classes:")
    for idx, name in enumerate(class_names):
        print(f"    - Class {idx}: '{name}'")

    if not class_names:
        print("[!] Error: No fields with valid bounding boxes found in any JSON label file.")
        return

    # Pass 2: Pair images, calculate normalized bboxes, and write YOLO labels
    print("\n[*] Pass 2: Matching JSON labels with images & calculating bounding boxes...")
    records = []
    missing_count = 0
    sample_image_names = [os.path.basename(p) for p in all_images[:5]]

    for j_path in json_files:
        try:
            with open(j_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # Match corresponding card image using clean stem (.jpg priority)
        img_path = find_matched_image(j_path, data, images_lookup, images_dir)
        if not img_path or not os.path.isfile(img_path):
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

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            ref_w, ref_h = get_coordinate_reference(data, img_w, img_h)
            scale_x = img_w / float(ref_w) if ref_w > 0 else 1.0
            scale_y = img_h / float(ref_h) if ref_h > 0 else 1.0

            boxes = []
            for fld in data.get("fields", []):
                fname = str(fld.get("name", "")).strip()
                if fname not in class_to_id:
                    continue

                pts = extract_field_points(fld)
                if not pts or len(pts) < 2:
                    continue

                xs = [p[0] * scale_x for p in pts]
                ys = [p[1] * scale_y for p in pts]

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
                nw = bw / float(img_w)
                nh = bh / float(img_h)

                # Clamp to [0, 1] range safely
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                boxes.append(f"{class_to_id[fname]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            if boxes:
                records.append((img_path, boxes))

        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}")

    print(f"\n[+] Successfully paired {len(records)} full card images with field bounding boxes.")
    if missing_count > 0:
        print(f"    - Unmatched images: {missing_count} (out of {len(json_files)} labels)")

    if not records:
        print("[!] No valid samples could be prepared. Please check the logs above.")
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
            # Copy image
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

    lbl_dir, img_dir = resolve_dataset_paths(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir
    )

    if not lbl_dir or not img_dir or not os.path.isdir(lbl_dir) or not os.path.isdir(img_dir):
        print(f"[!] Error: Could not resolve valid labels and images directories.")
        print(f"    Labels Dir: {lbl_dir}")
        print(f"    Images Dir: {img_dir}")
        print("    Please specify --data_dir (or --labels_dir and --images_dir)")
        sys.exit(1)

    convert_to_yolo_format(
        labels_dir=lbl_dir,
        images_dir=img_dir,
        output_dir=args.output_dir,
        val_split=args.val_split
    )
