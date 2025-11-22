import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np


def train_one_epoch_class(model, dataloader, criterion, optimizer, scheduler, device, cfg):
    """
    Train classification model for one epoch with gradient clipping.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Handle text input based on model type
        if cfg.MODEL.NAME.startswith("resnet_lstm"):
            text_ids = batch['text_ids'].to(device)
            logits = model(images, text_ids)
        elif cfg.MODEL.NAME.startswith("clip_gpt"):
            text_tokens = batch['text_tokens'].to(device)
            logits = model(images, text_tokens)
        elif cfg.MODEL.NAME.startswith("biovil_t"):
            texts = batch['text']
            logits = model(images, texts)
        else:
            raise ValueError(f"Unknown model: {cfg.MODEL.NAME}")

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # ADD GRADIENT CLIPPING
        if hasattr(cfg.SOLVER, 'GRADIENT_CLIP') and cfg.SOLVER.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.SOLVER.GRADIENT_CLIP)
        
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate_classification(model, dataloader, criterion, device, cfg):
    """
    Evaluate classification model.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Handle text input based on model type
            if cfg.MODEL.NAME.startswith("resnet_lstm"):
                text_ids = batch['text_ids'].to(device)
                logits = model(images, text_ids)
            elif cfg.MODEL.NAME.startswith("clip_gpt"):
                text_tokens = batch['text_tokens'].to(device)
                logits = model(images, text_tokens)
            elif cfg.MODEL.NAME.startswith("biovil_t"):
                texts = batch['text']
                logits = model(images, texts)
            else:
                raise ValueError(f"Unknown model: {cfg.MODEL.NAME}")

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Generate classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        output_dict=True,
        zero_division=0
    )

    return avg_loss, accuracy, f1, report


def train_one_epoch_gen(model, dataloader, criterion, optimizer, scheduler, device, cfg):
    """
    Train generation model for one epoch with gradient clipping.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training Generation", leave=False):
        images = batch['image'].to(device)
        
        # For ResNet-LSTM
        if cfg.MODEL.NAME == "resnet_lstm_generation":
            text_ids = batch['text_ids'].to(device)
            lengths = batch['lengths'].to(device)
            
            # Forward pass
            outputs = model(images, text_ids, lengths)
            
            # Compute loss (excluding padding)
            targets = text_ids[:, 1:]  # Skip <sos>
            outputs = outputs[:, :-1, :]  # Match dimensions
            
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
        
        # For CLIP-GPT2 and BioViL-T
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass (loss computed inside model)
            loss, _ = model(images, input_ids, attention_mask)

        optimizer.zero_grad()
        loss.backward()
        
        # ADD GRADIENT CLIPPING
        if hasattr(cfg.SOLVER, 'GRADIENT_CLIP') and cfg.SOLVER.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.SOLVER.GRADIENT_CLIP)
        
        optimizer.step()

        total_loss += loss.item()

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate_generation(model, dataloader, device, cfg, lstm_vocab=None):
    """
    Evaluate generation model with expanded metrics:
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - ROUGE-1, ROUGE-2, ROUGE-L
    - METEOR (NEW)
    """
    model.eval()
    
    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Generation", leave=False):
            images = batch['image'].to(device)
            raw_texts = batch['raw_texts']

            
            # Generate predictions based on model type
            if cfg.MODEL.NAME == "resnet_lstm_generation":
                # For LSTM, pass vocab and let model generate text directly
                generated_texts = model.generate(
                    images, 
                    max_len=cfg.DATA.MAX_SEQ_LEN,
                    vocab=lstm_vocab
                )
            
            else:
              from transformers import GPT2Tokenizer
              tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
              tokenizer.pad_token = tokenizer.eos_token

              if cfg.MODEL.NAME == "clip_gpt_generation":
                  generated_texts = model.generate(images, tokenizer, max_gen_len=cfg.DATA.MAX_SEQ_LEN)
              elif cfg.MODEL.NAME == "biovil_t_generation":
                  generated_texts = model.generate(images, tokenizer, max_gen_len=cfg.DATA.MAX_SEQ_LEN)
              else:
                  raise ValueError(f"Unknown generation model: {cfg.MODEL.NAME}")


            # Store references and hypotheses
            for ref_text, hyp_text in zip(raw_texts, generated_texts):
                all_references.append([ref_text.lower().split()])
                all_hypotheses.append(hyp_text.lower().split())

    # Compute BLEU scores (1-4)
    smoothing = SmoothingFunction().method1
    bleu_1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu_4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for ref, hyp in zip(all_references, all_hypotheses):
        ref_str = ' '.join(ref[0])
        hyp_str = ' '.join(hyp)
        scores = scorer.score(ref_str, hyp_str)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    rouge_1 = np.mean(rouge_1_scores)
    rouge_2 = np.mean(rouge_2_scores)
    rouge_l = np.mean(rouge_l_scores)

    # Compute METEOR scores (NEW)
    meteor_scores = []
    for ref, hyp in zip(all_references, all_hypotheses):
        # METEOR expects reference as list of tokens, hypothesis as list of tokens
        score = meteor_score(ref, hyp)
        meteor_scores.append(score)
    
    meteor = np.mean(meteor_scores)

    return {
        'BLEU-1': bleu_1,
        'BLEU-2': bleu_2,
        'BLEU-3': bleu_3,
        'BLEU-4': bleu_4,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l,
        'METEOR': meteor,  
    }
