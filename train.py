import argparse
import os
import json
import time
import pandas as pd
import torch
import pickle
from transformers import GPT2Tokenizer 

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from radmultibench.config import setup_cfg
from radmultibench.data import build_dataloaders
from radmultibench.models import build_model
from radmultibench.losses import build_criterion
from radmultibench.optim import build_optimizer, build_scheduler
from radmultibench.utils import setup_seed, setup_logger, extract_category, CATEGORY_MAPPING
from radmultibench.engine import (
    train_one_epoch_class,
    evaluate_classification,
    train_one_epoch_gen,
    evaluate_generation,
)


def compute_class_distribution(cfg):
    """
    Compute class distribution from dataset for loss weighting.
    
    Returns:
        class_counts: list of counts per class
    """
    df = pd.read_csv(cfg.DATA.CSV_PATH)
    
    # Pre-process dataframe
    df['Category'] = df['Image'].apply(extract_category)
    df = df[df['Category'] != 'Other']
    df['Category_Label'] = df['Category'].map(CATEGORY_MAPPING)
    
    # Get class distribution
    class_counts = df['Category_Label'].value_counts().sort_index().values
    
    return class_counts


def main(args):
    # 1. Setup Config, Logger, and Seed
    cfg = setup_cfg(args)
    logger = setup_logger(cfg.OUTPUT_DIR)
    setup_seed(cfg.SEED)

    logger.info(f"Starting experiment: {cfg.MODEL.NAME}")
    logger.info(f"Task: {cfg.TASK}")
    logger.info(f"Using device: {cfg.DEVICE}")
    logger.info(f"Config:\n{cfg}")

    device = torch.device(cfg.DEVICE)

    # 2. Compute Class Distribution (for loss weighting)
    class_counts = None
    if cfg.TASK == "classification":
        logger.info("Computing class distribution for loss weighting...")
        class_counts = compute_class_distribution(cfg)
        logger.info(f"Class distribution: {class_counts}")

    # 3. Build DataLoaders
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loader, vocab = build_dataloaders(cfg)
 
    if vocab:
        logger.info(f"Vocabulary built: {len(vocab)} unique tokens.")   

        vocab_path = os.path.join(cfg.OUTPUT_DIR, "vocab.pkl")
        logger.info(f"Saving vocabulary to {vocab_path}...")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    # 4. Build Model, Criterion, Optimizer, Scheduler
    logger.info(f"Building model: {cfg.MODEL.NAME}")
    model = build_model(cfg, vocab).to(device)

    # Build criterion with class weights for classification
    criterion = build_criterion(cfg, class_counts=class_counts)
    
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Print model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params / 1e6:.2f}M")

    # 5. Training Loop
    logger.info("Starting training loop...")

    best_metric = 0.0
    history = []

    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        start_time = time.time()

        if cfg.TASK == "classification":
            train_loss, train_acc = train_one_epoch_class(
                model, train_loader, criterion, optimizer, scheduler, device, cfg
            )
            val_loss, val_acc, val_f1, _ = evaluate_classification(
                model, val_loader, criterion, device, cfg
            )

            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch + 1}/{cfg.SOLVER.NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
            )

            metric = val_f1 

        else:  # generation
            train_loss = train_one_epoch_gen(
                model, train_loader, criterion, optimizer, scheduler, device, cfg
            )
            lstm_vocab = vocab if cfg.MODEL.NAME.startswith("resnet_lstm") else None
            
            # --- NEW METRICS LOGGING ---
            metrics = evaluate_generation(
                model, val_loader, device, cfg, lstm_vocab
            )
            
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{cfg.SOLVER.NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f}"
            )
            logger.info(
                f"  Val BLEU: B1={metrics['BLEU-1']:.4f}, B2={metrics['BLEU-2']:.4f}, "
                f"B3={metrics['BLEU-3']:.4f}, B4={metrics['BLEU-4']:.4f}"
            )
            logger.info(
                f"  Val ROUGE: R1={metrics['ROUGE-1']:.4f}, R2={metrics['ROUGE-2']:.4f}, "
                f"RL={metrics['ROUGE-L']:.4f}"
            )
            logger.info(
                f"  Val METEOR: {metrics['METEOR']:.4f}" 
            )
            
            metric = metrics['BLEU-4']  # Using BLEU-4 as the checkpointing metric

        # Getting current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_metric": metric,
                "lr": current_lr,
            }
        )

        # 6. Checkpointing
        if metric >= best_metric:
        # ===========================================
            best_metric = metric
            model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(
                f"✓ New best model saved to {model_path} (Metric: {best_metric:.4f})"
            )

    # 7. Final Evaluation on Test Set
    logger.info("=" * 80)
    logger.info("Training complete. Loading best model for final test evaluation...")
    logger.info("=" * 80)

    # Load best model
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))

    if cfg.TASK == "classification":
        test_loss, test_acc, test_f1, test_report = evaluate_classification(
            model, test_loader, criterion, device, cfg
        )
        
        logger.info("=" * 80)
        logger.info("FINAL TEST RESULTS - CLASSIFICATION")
        logger.info("=" * 80)
        logger.info(f"Test Loss:     {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc * 100:.2f}%")
        logger.info(f"Test F1-Macro: {test_f1:.4f}")
        logger.info("=" * 80)
        logger.info("Per-Class Classification Report:")
        logger.info("=" * 80)
        logger.info(f"{json.dumps(test_report, indent=2)}")
        logger.info("=" * 80)
        
        # Save results to JSON
        results = {
            "model": cfg.MODEL.NAME,
            "task": cfg.TASK,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
            "classification_report": test_report,
            "best_val_metric": float(best_metric),
            "num_epochs": cfg.SOLVER.NUM_EPOCHS,
            "num_params_M": num_params / 1e6,
        }
        
        results_path = os.path.join(cfg.OUTPUT_DIR, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    else:  # generation
        lstm_vocab = vocab if cfg.MODEL.NAME.startswith("resnet_lstm") else None
        
        # --- NEW FINAL LOGGING ---
        test_metrics = evaluate_generation(
            model, test_loader, device, cfg, lstm_vocab
        )
        
        logger.info("=" * 80)
        logger.info("FINAL TEST RESULTS - REPORT GENERATION")
        logger.info("=" * 80)
        logger.info("BLEU Scores:")
        logger.info(f"  BLEU-1:  {test_metrics['BLEU-1']:.4f}")
        logger.info(f"  BLEU-2:  {test_metrics['BLEU-2']:.4f}")
        logger.info(f"  BLEU-3:  {test_metrics['BLEU-3']:.4f}")
        logger.info(f"  BLEU-4:  {test_metrics['BLEU-4']:.4f}")
        logger.info("-" * 80)
        logger.info("ROUGE Scores:")
        logger.info(f"  ROUGE-1: {test_metrics['ROUGE-1']:.4f}")
        logger.info(f"  ROUGE-2: {test_metrics['ROUGE-2']:.4f}")
        logger.info(f"  ROUGE-L: {test_metrics['ROUGE-L']:.4f}")
        logger.info("-" * 80)
        logger.info(f"  METEOR:  {test_metrics['METEOR']:.4f}") 
        logger.info("=" * 80)

        
        # Save results to JSON
        results = {
            "model": cfg.MODEL.NAME,
            "task": cfg.TASK,
            "bleu_1": float(test_metrics['BLEU-1']),
            "bleu_2": float(test_metrics['BLEU-2']),
            "bleu_3": float(test_metrics['BLEU-3']),
            "bleu_4": float(test_metrics['BLEU-4']),
            "rouge_1": float(test_metrics['ROUGE-1']),
            "rouge_2": float(test_metrics['ROUGE-2']),
            "rouge_l": float(test_metrics['ROUGE-L']),
            "meteor": float(test_metrics['METEOR']),  
            "best_val_metric": float(best_metric),
            "num_epochs": cfg.SOLVER.NUM_EPOCHS,
            "num_params_M": num_params / 1e6,
        }
        
        results_path = os.path.join(cfg.OUTPUT_DIR, "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    # 8. Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(cfg.OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("=" * 80)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RadMultiBench Model Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="Path to the config.yaml file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if not args.config_file:
        raise ValueError("You must specify a --config-file.")

    main(args)
