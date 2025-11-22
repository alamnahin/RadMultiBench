import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import clip
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import random

# BioViL-T imports
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

from .utils import extract_category, CATEGORY_MAPPING

# -----------------------------------------------
# SECTION 1: VOCABULARY (For LSTM Models)
# -----------------------------------------------

class Vocab:
    """Vocabulary for text processing (LSTM models)."""
    def __init__(self, min_freq, pad_idx=0, unk_idx=1, sos_idx=2, eos_idx=3):
        self.word2idx = {'<pad>': pad_idx, '<unk>': unk_idx, '<sos>': sos_idx, '<eos>': eos_idx}
        self.idx2word = {pad_idx: '<pad>', unk_idx: '<unk>', sos_idx: '<sos>', eos_idx: '<eos>'}
        self.word_counts = Counter()
        self.min_freq = min_freq
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def add_sentence(self, sentence):
        tokens = word_tokenize(str(sentence).lower())
        self.word_counts.update(tokens)

    def build_vocab(self):
        idx = len(self.word2idx)
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2idx)

def build_vocab(dataframe, min_freq):
    vocab = Vocab(min_freq)
    for impression in dataframe['Impression']:
        vocab.add_sentence(impression)
    vocab.build_vocab()
    return vocab

# -----------------------------------------------
# SECTION 2: IMAGE TRANSFORMS (IMPROVED)
# -----------------------------------------------

def get_resnet_transforms(mode='train'):
    """Transforms for ResNet models with improved augmentation."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# -----------------------------------------------
# SECTION 3: TEXT AUGMENTATION
# -----------------------------------------------

def augment_medical_text(text, prob=0.3):
    """
    Simple text augmentation for short medical reports.
    Adds medical prefixes to increase report length and diversity.
    """
    if random.random() > prob or len(text.split()) > 15:
        return text
    
    prefixes = [
        "Radiological findings show ",
        "Impression suggests ",
        "Clinical evaluation reveals ",
        "Diagnostic assessment indicates ",
        "Imaging demonstrates ",
        "Examination shows ",
    ]
    
    return random.choice(prefixes) + text.lower()

# -----------------------------------------------
# SECTION 4: DATASETS (WITH AUGMENTATION)
# -----------------------------------------------

class BaseRadDataset(Dataset):
    """Base class to handle image loading and errors."""
    def __init__(self, dataframe, img_dir, transform, augment_text=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.augment_text = augment_text

    def __len__(self):
        return len(self.df)

    def get_image(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['Image'])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black') # Fallback
        return self.transform(image)
        
    def get_biovil_image(self, idx):
        """Image loader for BioViL-T (requires Grayscale)."""
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['Image'])
        try:
            image = Image.open(img_name).convert('L') # 'L' for grayscale
        except:
            image = Image.new('L', (512, 512), color=0) # Fallback
        return self.transform(image)
    
    def apply_text_augmentation(self, impression):
        """Apply text augmentation if enabled."""
        if self.augment_text:
            return augment_medical_text(impression, prob=0.3)
        return impression

# --- ResNet-LSTM Datasets ---

class RadMultiBenchLSTMDataset(BaseRadDataset):
    """Dataset for ResNet-LSTM Classification (Task 2)."""
    def __init__(self, dataframe, img_dir, transform, vocab, max_len, augment_text=False):
        super().__init__(dataframe, img_dir, transform, augment_text)
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        image = self.get_image(idx)
        label = self.df.iloc[idx]['Category_Label']
        impression = self.df.iloc[idx]['Impression']
        
        # Apply text augmentation
        impression = self.apply_text_augmentation(impression)
        
        tokens = word_tokenize(str(impression).lower())
        token_ids = [self.vocab.word2idx.get(t, self.vocab.unk_idx) for t in tokens]
        
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
            
        return {
            'image': image,
            'label': label,
            'text_ids': torch.tensor(token_ids, dtype=torch.long)
        }

class RadGenerationLSTMDataset(BaseRadDataset):
    """Dataset for ResNet-LSTM Generation (Task 1)."""
    def __init__(self, dataframe, img_dir, transform, vocab, max_len, augment_text=False):
        super().__init__(dataframe, img_dir, transform, augment_text)
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        image = self.get_image(idx)
        impression = self.df.iloc[idx]['Impression']
        
        # Apply text augmentation
        impression = self.apply_text_augmentation(impression)
        
        tokens = word_tokenize(str(impression).lower())
        token_ids = [self.vocab.sos_idx]
        token_ids.extend([self.vocab.word2idx.get(t, self.vocab.unk_idx) for t in tokens])
        token_ids.append(self.vocab.eos_idx)
        
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len-1] + [self.vocab.eos_idx]
            
        return {
            'image': image,
            'text_ids': torch.tensor(token_ids, dtype=torch.long),
            'raw_text': impression
        }

# --- CLIP-GPT Datasets ---

class RadMultiBenchCLIPDataset(BaseRadDataset):
    """Dataset for CLIP-GPT Classification (Task 2)."""
    def __init__(self, dataframe, img_dir, clip_preprocess, augment_text=False):
        super().__init__(dataframe, img_dir, clip_preprocess, augment_text)
        self.context_length = 77

    def __getitem__(self, idx):
        image_tensor = self.get_image(idx)
        label = self.df.iloc[idx]['Category_Label']
        impression = self.df.iloc[idx]['Impression']
        
        # Apply text augmentation
        impression = self.apply_text_augmentation(impression)
        
        text_tokens = clip.tokenize(impression, truncate=True).squeeze(0)
        
        return {
            'image': image_tensor,
            'label': label,
            'text_tokens': text_tokens
        }

class RadGenerationGPTDataset(BaseRadDataset):
    """Dataset for CLIP-GPT Generation (Task 1)."""
    def __init__(self, dataframe, img_dir, clip_preprocess, tokenizer, max_len, augment_text=False):
        super().__init__(dataframe, img_dir, clip_preprocess, augment_text)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        image_tensor = self.get_image(idx)
        impression = self.df.iloc[idx]['Impression']
        
        # Apply text augmentation
        impression = self.apply_text_augmentation(impression)
        
        text_encoding = self.tokenizer(
            impression,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image_tensor,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'raw_text': impression
        }

# --- BioViL-T Datasets ---

class BioViLTDataset(BaseRadDataset):
    """Dataset for BioViL-T Classification (Task 2)."""
    def __init__(self, dataframe, img_dir, image_transform, augment_text=False):
        super().__init__(dataframe, img_dir, image_transform, augment_text)

    def __getitem__(self, idx):
        image_tensor = self.get_biovil_image(idx)
        label = self.df.iloc[idx]['Category_Label']
        text = self.df.iloc[idx]['Impression']
        
        # Apply text augmentation
        text = self.apply_text_augmentation(text)
        
        return {
            'image': image_tensor,
            'text': text,
            'label': label
        }

class BioViLTGenerationDataset(BaseRadDataset):
    """Dataset for BioViL-T Generation (Task 1)."""
    def __init__(self, dataframe, img_dir, image_transform, tokenizer, max_len, augment_text=False):
        super().__init__(dataframe, img_dir, image_transform, augment_text)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        image_tensor = self.get_biovil_image(idx)
        impression = self.df.iloc[idx]['Impression']
        
        # Apply text augmentation
        impression = self.apply_text_augmentation(impression)
        
        text_encoding = self.tokenizer(
            impression,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image_tensor,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'raw_text': impression
        }

# -----------------------------------------------
# SECTION 5: COLLATE FUNCTIONS
# -----------------------------------------------

class LSTMCollateFn:
    """Pads sequences for LSTM models."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        text_ids_list = [item['text_ids'] for item in batch]
        text_ids_padded = pad_sequence(text_ids_list, batch_first=True, padding_value=self.pad_idx)
        
        return {
            'image': images,
            'label': labels,
            'text_ids': text_ids_padded
        }

class LSTMGenCollateFn:
    """Pads sequences for LSTM generation."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        raw_texts = [item['raw_text'] for item in batch]
        
        text_ids_list = [item['text_ids'] for item in batch]
        text_ids_padded = pad_sequence(text_ids_list, batch_first=True, padding_value=self.pad_idx)
        lengths = torch.tensor([len(seq) for seq in text_ids_list], dtype=torch.long)
        
        return {
            'image': images,
            'text_ids': text_ids_padded,
            'lengths': lengths,
            'raw_texts': raw_texts
        }

def gpt_collate_fn(batch):
    """Collate for GPT models (tensors are already padded by tokenizer)."""
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    raw_texts = [item['raw_text'] for item in batch]
    
    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'raw_texts': raw_texts
    }

def biovil_class_collate_fn(batch):
    """Collate for BioViL-T classification (text is processed in model)."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    texts = [item['text'] for item in batch]
    
    return {
        'image': images,
        'text': texts,
        'label': labels
    }
    
def biovil_gen_collate_fn(batch):
    """Collate for BioViL-T generation (same as GPT)."""
    return gpt_collate_fn(batch)

# -----------------------------------------------
# SECTION 6: BALANCED SAMPLING
# -----------------------------------------------

def get_balanced_sampler(labels):
    """
    Create weighted sampler for balanced training.
    Args:
        labels: numpy array or list of class labels
    Returns:
        WeightedRandomSampler
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    print(f"[INFO] Class counts: {class_counts}")
    print(f"[INFO] Class weights: {class_weights}")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# -----------------------------------------------
# SECTION 7: MAIN DATALOADER BUILDER
# -----------------------------------------------

def build_dataloaders(cfg):
    """
    Main factory function to build and return data loaders and vocab.
    Now includes class balancing for training set.
    """
    df = pd.read_csv(cfg.DATA.CSV_PATH)
    
    # Pre-process dataframe
    df['Category'] = df['Image'].apply(extract_category)
    df = df[df['Category'] != 'Other']
    df['Category_Label'] = df['Category'].map(CATEGORY_MAPPING)

    # Split: 70/15/15
    train_df, temp_df = train_test_split(
        df, 
        test_size=(cfg.DATA.SPLIT_RATIOS[1] + cfg.DATA.SPLIT_RATIOS[2]), 
        random_state=cfg.SEED,
        stratify=df['Category_Label']
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(cfg.DATA.SPLIT_RATIOS[2] / (cfg.DATA.SPLIT_RATIOS[1] + cfg.DATA.SPLIT_RATIOS[2])), 
        random_state=cfg.SEED,
        stratify=temp_df['Category_Label']
    )

    print(f"[INFO] Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    model_name = cfg.MODEL.NAME
    task = cfg.TASK
    img_dir = cfg.DATA.IMG_DIR
    batch_size = cfg.SOLVER.BATCH_SIZE
    num_workers = cfg.DATA.NUM_WORKERS
    
    # Initialize variables
    train_dataset, val_dataset, test_dataset = None, None, None
    collate_fn = None
    vocab = None
    train_sampler = None
    
    if model_name.startswith("resnet_lstm"):
        # Build vocab from training data
        vocab = build_vocab(train_df, min_freq=cfg.DATA.MIN_WORD_FREQ)
        transform = get_resnet_transforms('train')
        val_transform = get_resnet_transforms('val')
        
        if task == "classification":
            train_dataset = RadMultiBenchLSTMDataset(
                train_df, img_dir, transform, vocab, cfg.DATA.MAX_SEQ_LEN, augment_text=True
            )
            val_dataset = RadMultiBenchLSTMDataset(
                val_df, img_dir, val_transform, vocab, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            test_dataset = RadMultiBenchLSTMDataset(
                test_df, img_dir, val_transform, vocab, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            collate_fn = LSTMCollateFn(pad_idx=vocab.pad_idx)
            
            # Create balanced sampler
            train_labels = train_df['Category_Label'].values
            train_sampler = get_balanced_sampler(train_labels)
            
        else: # generation
            train_dataset = RadGenerationLSTMDataset(
                train_df, img_dir, transform, vocab, cfg.DATA.MAX_SEQ_LEN, augment_text=True
            )
            val_dataset = RadGenerationLSTMDataset(
                val_df, img_dir, val_transform, vocab, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            test_dataset = RadGenerationLSTMDataset(
                test_df, img_dir, val_transform, vocab, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            collate_fn = LSTMGenCollateFn(pad_idx=vocab.pad_idx)

    elif model_name.startswith("clip_gpt"):
        _, clip_preprocess = clip.load("ViT-B/32", device="cpu")
        
        if task == "classification":
            train_dataset = RadMultiBenchCLIPDataset(train_df, img_dir, clip_preprocess, augment_text=True)
            val_dataset = RadMultiBenchCLIPDataset(val_df, img_dir, clip_preprocess, augment_text=False)
            test_dataset = RadMultiBenchCLIPDataset(test_df, img_dir, clip_preprocess, augment_text=False)
            collate_fn = None
            
            # Create balanced sampler
            train_labels = train_df['Category_Label'].values
            train_sampler = get_balanced_sampler(train_labels)
            
        else: # generation
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            train_dataset = RadGenerationGPTDataset(
                train_df, img_dir, clip_preprocess, tokenizer, cfg.DATA.MAX_SEQ_LEN, augment_text=True
            )
            val_dataset = RadGenerationGPTDataset(
                val_df, img_dir, clip_preprocess, tokenizer, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            test_dataset = RadGenerationGPTDataset(
                test_df, img_dir, clip_preprocess, tokenizer, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            collate_fn = gpt_collate_fn

    elif model_name.startswith("biovil_t"):
        img_transform = create_chest_xray_transform_for_inference(resize=512, center_crop_size=480)
        
        if task == "classification":
            train_dataset = BioViLTDataset(train_df, img_dir, img_transform, augment_text=True)
            val_dataset = BioViLTDataset(val_df, img_dir, img_transform, augment_text=False)
            test_dataset = BioViLTDataset(test_df, img_dir, img_transform, augment_text=False)
            collate_fn = biovil_class_collate_fn
            
            # Create balanced sampler
            train_labels = train_df['Category_Label'].values
            train_sampler = get_balanced_sampler(train_labels)
            
        else: # generation
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            train_dataset = BioViLTGenerationDataset(
                train_df, img_dir, img_transform, tokenizer, cfg.DATA.MAX_SEQ_LEN, augment_text=True
            )
            val_dataset = BioViLTGenerationDataset(
                val_df, img_dir, img_transform, tokenizer, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            test_dataset = BioViLTGenerationDataset(
                test_df, img_dir, img_transform, tokenizer, cfg.DATA.MAX_SEQ_LEN, augment_text=False
            )
            collate_fn = biovil_gen_collate_fn
            
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Create DataLoaders
    # Train loader: use sampler if available (for classification), otherwise shuffle
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  
        shuffle=(train_sampler is None),  
        num_workers=num_workers, 
        collate_fn=collate_fn, 
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, vocab
