import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------------------------
# MODEL 1: CLIP-GPT (Task 2 - Classification)
# -----------------------------------------------

class MultimodalCLIPClassifier(nn.Module):
    
    def __init__(self, num_classes=5, dropout=0.1, clip_dim=512):
        super(MultimodalCLIPClassifier, self).__init__()

        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_dim = clip_dim
        
        for param in self.clip_model.parameters():
            param.requires_grad = False

        fused_dim = self.clip_dim * 2 

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_tokens, mode="vision+text"):
      with torch.no_grad():
          image_features = self.clip_model.encode_image(images).float()
          text_features = self.clip_model.encode_text(text_tokens).float()

      if mode == "vision+text":
          fused_features = torch.cat([image_features, text_features], dim=1)
      elif mode == "vision_only":
          fused_features = torch.cat([image_features, torch.zeros_like(text_features)], dim=1)
      elif mode == "text_only":
          fused_features = torch.cat([torch.zeros_like(image_features), text_features], dim=1)

      logits = self.classifier(fused_features)
      return logits

# -----------------------------------------------
# MODEL 2: CLIP-GPT (Task 1 - Generation)
# -----------------------------------------------

class CLIPGPTGenerationModel(nn.Module):
    
    def __init__(self, clip_dim=512, gpt_dim=768, dropout=0.1, prefix_length=1):
        super(CLIPGPTGenerationModel, self).__init__()

        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_dim = clip_dim
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = gpt_dim
        self.prefix_length = prefix_length

        self.vision_projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.gpt_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, images, input_ids, attention_mask):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
            
        image_prefix = self.vision_projection(image_features).unsqueeze(1)
        text_embeds = self.gpt_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_prefix, text_embeds], dim=1)
        
        prefix_mask = torch.ones(images.size(0), self.prefix_length, dtype=torch.long).to(images.device)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        
        # Create a copy of input_ids for labels
        text_labels = input_ids.clone()
        # Set label to -100 wherever attention_mask is 0 (padding)
        text_labels[attention_mask == 0] = -100
        # Create labels, masking the prefix
        prefix_labels = torch.full((images.size(0), self.prefix_length), -100, dtype=torch.long).to(images.device)
        labels = torch.cat([prefix_labels, text_labels], dim=1)

        outputs = self.gpt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

    @torch.no_grad()
    def generate(self, images, tokenizer, max_gen_len=60):
        self.eval()
        batch_size = images.size(0) 
        device = images.device     
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
            image_prefix = self.vision_projection(image_features).unsqueeze(1)

        
        attention_mask = torch.ones(batch_size, self.prefix_length, dtype=torch.long).to(device)
        

        generated_ids = self.gpt_model.generate(
            inputs_embeds=image_prefix,
            attention_mask=attention_mask, 
            max_length=max_gen_len + self.prefix_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=4,
            early_stopping=True
        )
        
        # Remove prefix
        generated_ids = generated_ids[:, self.prefix_length:]
        
        generated_texts = []
        for ids in generated_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
