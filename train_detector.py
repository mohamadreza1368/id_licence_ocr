import os
import sys
import argparse
import shutil
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def train_field_detector(
    data_yaml: str = "detector_data/dataset.yaml",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = None
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[!] Error: 'ultralytics' is not installed.")
        print("    Please install it using: pip install ultralytics")
        return

    data_yaml = os.path.abspath(data_yaml)
    if not os.path.exists(data_yaml):
        print(f"[!] Error: Dataset configuration file not found: {data_yaml}")
        print("    Please run 'python prepare_detector_data.py' first.")
        return

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    if device is None:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  Iranian National ID Card - Field Detector Training (YOLO-Nano)")
    print("=" * 65)
    print(f"[*] Dataset Config : {data_yaml}")
    print(f"[*] Device         : {device}")
    print(f"[*] Hyperparameters: Epochs={epochs} | Batch={batch_size} | ImgSize={imgsz}")

    # Load ultra-lightweight pretrained YOLOv8-nano (~3.2 MB)
    print("[*] Initializing YOLOv8-Nano backbone...")
    model = YOLO("yolov8n.pt")

    # Start training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project="runs/detector",
        name="id_field_detector",
        exist_ok=True,
        verbose=True
    )

    # Copy best weights to checkpoints directory
    best_weights = os.path.join("runs", "detector", "id_field_detector", "weights", "best.pt")
    target_weights = os.path.join(config.CHECKPOINTS_DIR, "best_field_detector.pt")

    if os.path.exists(best_weights):
        shutil.copy2(best_weights, target_weights)
        print(f"\n[+] Successfully saved best field detector to: {target_weights}")

        # Also export to ONNX for mobile deployment
        try:
            print("[*] Exporting detector to ONNX for mobile...")
            best_model = YOLO(target_weights)
            best_model.export(format="onnx", imgsz=imgsz)
            print("[+] Exported detector ONNX model successfully!")
        except Exception as e:
            print(f"[!] Note: ONNX export skipped: {e}")

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight YOLO-Nano Field Detector on Iranian ID cards")
    parser.add_argument("--data", type=str, default="detector_data/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('cpu', '0', etc.)")
    args = parser.parse_args()

    train_field_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device
    )
