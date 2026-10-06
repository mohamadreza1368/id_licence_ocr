import os
import torch
import config
from tokenizer import CTCLabelConverter
from model import MobileOCRNet

class ExportableOCRNet(torch.nn.Module):
    """
    Wrapper for mobile deployment: directly outputs greedy predictions [B, T]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [B, 3, 32, W]
        log_probs = self.model(x)  # [T, B, C]
        probs = log_probs.permute(1, 0, 2)  # [B, T, C]
        preds = torch.argmax(probs, dim=2)  # [B, T]
        return preds

def export_models(checkpoint_path=None):
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, "best_mobile_ocr.pth")

    converter = CTCLabelConverter(config.ALL_CHARS)
    model = MobileOCRNet(num_classes=converter.num_classes)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[*] Loaded trained weights from: {checkpoint_path}")
    else:
        print(f"[!] Warning: No checkpoint found at {checkpoint_path}. Exporting architecture for test.")

    model.eval()
    exportable = ExportableOCRNet(model)
    exportable.eval()

    dummy_input = torch.randn(1, 3, 32, 160)

    # 1. Export to TorchScript (PyTorch Mobile for Android / iOS)
    torchscript_path = os.path.join(config.CHECKPOINTS_DIR, "model_mobile.pt")
    try:
        traced_model = torch.jit.trace(exportable, dummy_input)
        traced_model.save(torchscript_path)
        size_mb = os.path.getsize(torchscript_path) / (1024 * 1024)
        print(f"[+] TorchScript mobile model exported: {torchscript_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[!] TorchScript export failed: {e}")

    # 2. Export to ONNX (if 'onnx' is installed)
    onnx_path = os.path.join(config.CHECKPOINTS_DIR, "model.onnx")
    try:
        import onnx
        torch.onnx.export(
            exportable,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 3: "width"},
                "output": {0: "batch_size", 1: "seq_len"}
            },
            dynamo=False
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[+] ONNX model exported: {onnx_path} ({size_mb:.2f} MB)")
    except ImportError:
        print("[*] Info: 'onnx' package not installed. TorchScript model is ready for mobile.")
        print("    (To also export ONNX, run: pip install onnx)")
    except Exception as e:
        print(f"[!] ONNX export error: {e}")

    # 3. Save Vocabulary
    vocab_path = os.path.join(config.CHECKPOINTS_DIR, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for idx, ch in enumerate(converter.character):
            f.write(f"{idx}\t{ch}\n")
    print(f"[+] Vocabulary mapping saved: {vocab_path}")

if __name__ == "__main__":
    export_models()
