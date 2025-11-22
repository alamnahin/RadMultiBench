import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# BioViL-T imports
from health_multimodal.image import ImageModel
from health_multimodal.text import get_bert_inference

# -----------------------------------------------
# MODEL 1: BioViL-T (Task 2 - Classification)
# -----------------------------------------------


class BioViLTClassifier(nn.Module):
    """BioViL-T for classification."""
    def __init__(self, num_classes=5, fused_dim=2176, dropout=0.1): 
        super(BioViLTClassifier, self).__init__()

        self.image_model = ImageModel(
            img_encoder_type="resnet50",
            joint_feature_size=128
        )
        self.text_inference = get_bert_inference()
        self.text_model = self.text_inference.model

        # Freeze encoders
        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # This is the global average pooling layer
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fused_dim is 2176 (2048 from image + 128 from text)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, texts, mode="vision+text"):
      with torch.no_grad():
          
          patch_features = self.image_model(images).patch_embeddings # [B, 2048, 7, 7]
          # Pool them to get [B, 2048]
          pooled_features = self.image_pool(patch_features)
          image_features = pooled_features.squeeze(-1).squeeze(-1) # [B, 2048]
          # ===========================================
          
          tokenized = self.text_inference.tokenizer(
              texts, padding=True, truncation=True,
              max_length=512, return_tensors='pt'
          ).to(images.device)
          text_features = self.text_model.get_projected_text_embeddings(
              input_ids=tokenized['input_ids'],
              attention_mask=tokenized['attention_mask'],
          ) # [B, 128]

      if mode == "vision+text":
          # This is now [B, 2048 + 128] = [B, 2176]
          fused = torch.cat([image_features, text_features], dim=1)
      elif mode == "vision_only":
          # This is now [B, 2048 + 0] = [B, 2176]
          fused = torch.cat([image_features, torch.zeros_like(text_features)], dim=1)
      elif mode == "text_only":
          # This is now [B, 0 + 128] = [B, 2176]
          # Note: We must create zeros with the correct image feature size
          image_zeros = torch.zeros_like(image_features)
          fused = torch.cat([image_zeros, text_features], dim=1)

      # This will now work, as `fused` is [B, 2176]
      logits = self.classifier(fused)
      return logits


# -----------------------------------------------
# MODEL 2: BioViL-T (Task 1 - Generation)
# -----------------------------------------------


class BioViLTGPTGenerator(nn.Module):
    """BioViL-T for Generation."""
    def __init__(self, image_dim=2048, gpt_dim=768, dropout=0.1, prefix_length=1): 
        super(BioViLTGPTGenerator, self).__init__()

        self.image_model = ImageModel(
            img_encoder_type="resnet50",
            joint_feature_size=128
        )
        for param in self.image_model.parameters():
            param.requires_grad = False
            
        self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_dim = gpt_dim
        self.prefix_length = prefix_length
        
        
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # image_dim is now 2048
        self.vision_projection = nn.Sequential(
            nn.Linear(image_dim, gpt_dim), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, images, input_ids, attention_mask):
        with torch.no_grad():
            
            patch_features = self.image_model(images).patch_embeddings
            # Pool them to get [B, 2048]
            pooled_features = self.image_pool(patch_features)
            image_features = pooled_features.squeeze(-1).squeeze(-1)
            # -------------------------
        
        image_prefix = self.vision_projection(image_features).unsqueeze(1) # [B, 1, 768]
        text_embeds = self.gpt_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([image_prefix, text_embeds], dim=1)
        
        prefix_mask = torch.ones(images.size(0), self.prefix_length, dtype=torch.long).to(images.device)
        combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Create a copy of input_ids for labels
        text_labels = input_ids.clone()
        # Set label to -100 wherever attention_mask is 0 (padding)
        text_labels[attention_mask == 0] = -100

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
            
            patch_features = self.image_model(images).patch_embeddings
            pooled_features = self.image_pool(patch_features)
            image_features = pooled_features.squeeze(-1).squeeze(-1) # [B, 2048]
            # ------------------------------
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
        
        generated_ids = generated_ids[:, self.prefix_length:]
        
        generated_texts = []
        for ids in generated_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
            
        return generated_texts
