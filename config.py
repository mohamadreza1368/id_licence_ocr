import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_LABEL_FILE = os.path.join(DATA_DIR, "train.txt")
VAL_LABEL_FILE = os.path.join(DATA_DIR, "val.txt")

# Image settings for OCR recognizer
IMG_HEIGHT = 32
MIN_WIDTH = 32
MAX_WIDTH = 320  # Max width to avoid excessive padding in batches

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Vocabulary: Persian alphabet, Persian digits, English digits, and special characters
# Blank token index for CTC is 0
PERSIAN_CHARS = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئءآأؤة"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ENGLISH_DIGITS = "0123456789"
SPECIAL_CHARS = " /-.:"

# All characters in vocab (CTC blank token will be prepended at index 0)
ALL_CHARS = PERSIAN_CHARS + PERSIAN_DIGITS + ENGLISH_DIGITS + SPECIAL_CHARS
