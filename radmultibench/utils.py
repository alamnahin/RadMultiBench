import torch
import numpy as np
import random
import os
import logging

CATEGORY_MAPPING = {
    "Breast Cancer": 0,
    "Lung Benign": 1,
    "Lung Malignant": 2,
    "Pneumonia": 3,
    "Tuberculosis": 4,
}

INV_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_category(image_name):
    """Extracts category from image filename."""
    img_lower = str(image_name).lower()
    if "breast cancer" in img_lower:
        return "Breast Cancer"
    if "lung bengin" in img_lower or "lung benign" in img_lower:
        return "Lung Benign"
    if "lung malignant" in img_lower:
        return "Lung Malignant"
    if "pneumonia" in img_lower:
        return "Pneumonia"
    if img_lower.startswith("tb.") or "tb." in img_lower:
        return "Tuberculosis"
    return "Other"


def setup_logger(output_dir, name="radmultibench"):
    """Sets up a logger to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
