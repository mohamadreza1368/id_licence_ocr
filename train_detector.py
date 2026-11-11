import os
import sys
import time
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
    device: str = None,
    use_mlflow: bool = config.MLFLOW_ENABLE,
    mlflow_uri: str = config.MLFLOW_TRACKING_URI,
    experiment_name: str = "iranian_id_card_detector",
    run_name: str = None
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

    # Setup MLflow Tracking if enabled
    active_run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
            os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
            actual_run_name = run_name or f"yolov8n_detector_{int(time.time())}"
            active_run = mlflow.start_run(run_name=actual_run_name)
            mlflow.log_params({
                "architecture": "YOLOv8-Nano",
                "epochs": epochs,
                "batch_size": batch_size,
                "imgsz": imgsz,
                "device": device
            })
            print(f"[+] Connected to MLflow Tracking Server: {mlflow_uri}")
            print(f"    - Experiment: {experiment_name} | Run: {actual_run_name}")
        except Exception as e:
            print(f"[!] Warning: Could not initialize MLflow ({e}). Continuing locally.")

    # Load model: try pretrained yolov8n.pt first; fallback to yolov8n.yaml if offline
    print("\n[*] Initializing YOLOv8-Nano backbone...")
    try:
        model = YOLO("yolov8n.pt")
        print("[+] Loaded pretrained yolov8n.pt backbone.")
    except Exception as e:
        print(f"[*] Pretrained weights download unavailable ({e}).")
        print("[*] Initializing YOLOv8-Nano architecture from scratch (yolov8n.yaml)...")
        model = YOLO("yolov8n.yaml")
        print("[+] YOLOv8-Nano model initialized successfully.")

    # Start training
    print("\n[*] Starting YOLO training loop...")
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
        onnx_file = None
        try:
            print("[*] Exporting detector to ONNX for mobile deployment...")
            best_model = YOLO(target_weights)
            exported = best_model.export(format="onnx", imgsz=imgsz)
            if exported and os.path.isfile(exported):
                onnx_target = os.path.join(config.CHECKPOINTS_DIR, "best_field_detector.onnx")
                shutil.copy2(exported, onnx_target)
                onnx_file = onnx_target
                print(f"[+] Exported detector ONNX model to: {onnx_target}")
        except Exception as e:
            print(f"[!] Note: ONNX export skipped: {e}")

        # Log weights to MLflow
        if active_run:
            try:
                import mlflow
                mlflow.log_artifact(target_weights, artifact_path="weights")
                if onnx_file and os.path.isfile(onnx_file):
                    mlflow.log_artifact(onnx_file, artifact_path="weights")
                mlflow.end_run()
                print("[+] Successfully logged model artifacts to MLflow.")
            except Exception as e:
                print(f"[!] Note: Could not upload artifacts to MLflow: {e}")

    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight YOLO-Nano Field Detector on Iranian ID cards")
    parser.add_argument("--data", type=str, default="detector_data/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('cpu', '0', etc.)")
    parser.add_argument("--use_mlflow", action="store_true", default=config.MLFLOW_ENABLE, help="Log metrics to MLflow")
    parser.add_argument("--mlflow_uri", type=str, default=config.MLFLOW_TRACKING_URI, help="MLflow Tracking Server URI")
    parser.add_argument("--experiment_name", type=str, default="iranian_id_card_detector", help="MLflow Experiment Name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow Run Name")
    args = parser.parse_args()

    train_field_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        use_mlflow=args.use_mlflow,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )
