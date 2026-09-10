"""
Central configuration for the prostate segmentation project.

All paths are overridable via environment variables so the same code runs
unchanged locally, on a lab server, or in Colab (with DATA_ROOT pointed at a
mounted Drive folder).
"""
import os

# --- Paths ---------------------------------------------------------------
DATA_ROOT = os.environ.get("PROSTATE_DATA_ROOT", "./data")
RAW_DATASET_DIR = os.path.join(DATA_ROOT, "Micro_Ultrasound_Prostate_Segmentation_Dataset")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed_512")
TEST_PROCESSED_DIR = os.path.join(DATA_ROOT, "processed_512_test")
EMBEDDING_CACHE_DIR = os.path.join(DATA_ROOT, "medsam_embedding_cache")
SPLIT_FILE = os.path.join(DATA_ROOT, "patient_split.json")
CHECKPOINT_DIR = os.environ.get("PROSTATE_CKPT_DIR", "./checkpoints")
RESULTS_DIR = os.environ.get("PROSTATE_RESULTS_DIR", "./results")

# Dataset source: Jiang et al., "MicroSegNet" (2024) — CC-BY-4.0
ZENODO_URL = (
    "https://zenodo.org/records/10475293/files/"
    "Micro_Ultrasound_Prostate_Segmentation_Dataset.zip?download=1"
)

# --- Image / preprocessing -------------------------------------------------
IMG_SIZE = 512

# --- Phase 1: UNet ---------------------------------------------------------
UNET_BATCH_SIZE = 4
UNET_LR = 1e-4
UNET_EPOCHS = 30
UNET_PATIENCE = 8
UNET_CKPT_NAME = "unet_best.pth"

# --- Phase 2: MedSAM / LoRA -------------------------------------------------
MEDSAM_MODEL_ID = "wanglab/medsam-vit-base"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_BATCH_SIZE = 1          # batch>1 OOMs on SAM's global attention at 1024x1024 — see README
LORA_GRAD_ACCUM_STEPS = 4    # effective batch size 4, one sample's activations in memory at a time
LORA_EPOCHS = 15
LORA_LR = 1e-4
LORA_PATIENCE = 6
LORA_CKPT_NAME = "medsam_lora_r8_best.pth"

for _dir in (DATA_ROOT, CHECKPOINT_DIR, RESULTS_DIR):
    os.makedirs(_dir, exist_ok=True)
