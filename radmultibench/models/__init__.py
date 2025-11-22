from .clip_gpt import MultimodalCLIPClassifier, CLIPGPTGenerationModel
from .resnet_lstm import ResNetLSTMModel, ResNetLSTMGenerationModel
from .biovil_t import BioViLTClassifier, BioViLTGPTGenerator


def build_model(cfg, vocab=None):
    """Factory function to build the model."""

    model_name = cfg.MODEL.NAME

    if model_name == "resnet_lstm_classification":
        if vocab is None:
            raise ValueError("ResNet-LSTM needs a vocab.")
        return ResNetLSTMModel(
            vocab_size=len(vocab),
            num_classes=cfg.MODEL.NUM_CLASSES,
            embed_dim=cfg.MODEL.EMBED_DIM,
            lstm_hidden=cfg.MODEL.LSTM_HIDDEN_DIM,
            dropout=cfg.MODEL.DROPOUT,
        )

    elif model_name == "resnet_lstm_generation":
        if vocab is None:
            raise ValueError("ResNet-LSTM needs a vocab.")
        return ResNetLSTMGenerationModel(
            vocab_size=len(vocab),
            embed_dim=cfg.MODEL.EMBED_DIM,
            lstm_hidden=cfg.MODEL.LSTM_HIDDEN_DIM,
            encoder_dim=cfg.MODEL.ENCODER_DIM,
            dropout=cfg.MODEL.DROPOUT,
            pad_idx=cfg.DATA.PAD_IDX,
            sos_idx=cfg.DATA.SOS_IDX,
            eos_idx=cfg.DATA.EOS_IDX,
        )

    elif model_name == "clip_gpt_classification":
        return MultimodalCLIPClassifier(
            num_classes=cfg.MODEL.NUM_CLASSES, dropout=cfg.MODEL.DROPOUT
        )

    elif model_name == "clip_gpt_generation":
        return CLIPGPTGenerationModel(
            clip_dim=cfg.MODEL.CLIP_DIM,
            gpt_dim=cfg.MODEL.GPT_DIM,
            dropout=cfg.MODEL.DROPOUT,
            prefix_length=cfg.MODEL.PREFIX_LENGTH,
        )

    elif model_name == "biovil_t_classification":
        return BioViLTClassifier(
            num_classes=cfg.MODEL.NUM_CLASSES,
            fused_dim=cfg.MODEL.FUSED_DIM,  # Pass the corrected dim
            dropout=cfg.MODEL.DROPOUT,
        )

    elif model_name == "biovil_t_generation":
        return BioViLTGPTGenerator(
            image_dim=cfg.MODEL.IMAGE_DIM,  # Pass the corrected dim
            gpt_dim=cfg.MODEL.GPT_DIM,
            dropout=cfg.MODEL.DROPOUT,
            prefix_length=cfg.MODEL.PREFIX_LENGTH,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")
