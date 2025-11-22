import os
import yaml  
from yacs.config import CfgNode as CN

_C = CN()
_C._BASE_ = ""

# -----------------------------------------------
# Base config
# -----------------------------------------------
_C.TASK = "classification"  # 'classification' or 'generation'

_C.DATA = CN()
_C.DATA.IMG_DIR = "./data/images"
_C.DATA.CSV_PATH = "./data/clean_map.csv"
_C.DATA.SPLIT_RATIOS = [0.7, 0.15, 0.15]  # Train, Val, Test
_C.DATA.NUM_WORKERS = 4
_C.DATA.MIN_WORD_FREQ = 3
_C.DATA.MAX_SEQ_LEN = 64
_C.DATA.PAD_IDX = 0
_C.DATA.SOS_IDX = 2
_C.DATA.EOS_IDX = 3

_C.MODEL = CN()
_C.MODEL.NAME = "resnet_lstm_classification"
_C.MODEL.DROPOUT = 0.1
_C.MODEL.NUM_CLASSES = 5
_C.MODEL.ABLATION_MODE = "vision+text"
# ResNet-LSTM
_C.MODEL.EMBED_DIM = 300
_C.MODEL.LSTM_HIDDEN_DIM = 512
_C.MODEL.ATTENTION_DIM = 256
_C.MODEL.ENCODER_DIM = 2048
# CLIP-GPT
_C.MODEL.CLIP_DIM = 512
_C.MODEL.GPT_DIM = 768
_C.MODEL.PREFIX_LENGTH = 1
# BioViL-T
_C.MODEL.IMAGE_DIM = 2048
_C.MODEL.FUSED_DIM = 256


_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "AdamW"
_C.SOLVER.LR = 5e-5
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.BETAS = [0.9, 0.999]
_C.SOLVER.WARMUP_STEPS = 500
_C.SOLVER.NUM_EPOCHS = 25
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LABEL_SMOOTHING = 0.1
_C.SOLVER.GRADIENT_CLIP = 1.0

_C.SEED = 42
_C.OUTPUT_DIR = "./output"
_C.DEVICE = "cuda"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()


def load_cfg_from_file(cfg_path: str):
    """Load config from a YAML file."""
    
    cfg = get_cfg_defaults()

    
    with open(cfg_path, 'r') as f:
        config_data = yaml.safe_load(f)

    
    if "_BASE_" in config_data:
        base_cfg_path = os.path.join(os.path.dirname(cfg_path), config_data["_BASE_"])
        cfg.merge_from_file(base_cfg_path)
    
    
    cfg.merge_from_file(cfg_path)
    
    
    if "_BASE_" in cfg:
        cfg.pop("_BASE_")

    return cfg


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    Loads from YAML, merges with command-line args, and freezes.
    """
    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)  

    
    model_name = cfg.MODEL.NAME
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, model_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    return cfg