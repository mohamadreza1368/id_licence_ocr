import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import config
from tokenizer import CTCLabelConverter

class PersianOCRDataset(Dataset):
    def __init__(self, label_file, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.samples = []

        if os.path.exists(label_file):
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = parts[1]
                        if os.path.exists(img_path):
                            self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} samples from {label_file}")

    def __len__(self):
        return len(self.samples)

    def preprocess_image(self, img: Image.Image):
        # Target height is 32
        w, h = img.size
        scale = config.IMG_HEIGHT / float(h)
        new_w = max(config.MIN_WIDTH, min(config.MAX_WIDTH, int(w * scale)))

        # Ensure new_w is divisible by 4 (since downsampling along width is 4x)
        if new_w % 4 != 0:
            new_w = (new_w // 4 + 1) * 4

        resized_img = img.resize((new_w, config.IMG_HEIGHT), Image.Resampling.BILINEAR)

        # Convert to tensor [3, H, W] in range [-1, 1]
        tensor = TF.to_tensor(resized_img)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return tensor

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_tensor = self.preprocess_image(img)
        return img_tensor, label


def ocr_collate_fn(batch, converter: CTCLabelConverter):
    """
    Collate function to batch variable-width images and prepare CTC targets.
    """
    images, labels = zip(*batch)

    # Find max width in this batch
    max_w = max(img.shape[2] for img in images)

    # Pad images on the right with 0 (which is 0.5 normalized)
    padded_images = []
    input_lengths = []
    for img in images:
        c, h, w = img.shape
        pad_w = max_w - w
        if pad_w > 0:
            padded = torch.nn.functional.pad(img, (0, pad_w, 0, 0), value=0.0)
        else:
            padded = img
        padded_images.append(padded)
        # Sequence length after 4x downsampling in the CNN
        input_lengths.append(max_w // 4)

    padded_images = torch.stack(padded_images, dim=0)

    # Encode labels for CTC
    encoded_targets, target_lengths = converter.encode(labels)

    targets_tensor = torch.tensor(encoded_targets, dtype=torch.long)
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    return padded_images, targets_tensor, input_lengths_tensor, target_lengths_tensor, labels
