import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------------------------
# MODEL 1: ResNet-LSTM (Task 2 - Classification)
# -----------------------------------------------

class ResNetLSTMModel(nn.Module):
    
    def __init__(self, vocab_size, num_classes=5, embed_dim=300, lstm_hidden=512, dropout=0.1):
        super(ResNetLSTMModel, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 2048 
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )

        fused_dim = self.feature_dim + lstm_hidden

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self.init_h = nn.Linear(self.feature_dim, lstm_hidden)
        self.init_c = nn.Linear(self.feature_dim, lstm_hidden)

    def forward(self, images, text_ids, mode="vision+text"):
      features = self.encoder(images)
      batch_size = features.size(0)
      pooled_features = self.gap(features).view(batch_size, -1) 

      h0 = self.init_h(pooled_features).unsqueeze(0)
      c0 = self.init_c(pooled_features).unsqueeze(0)
      text_embeds = self.embedding(text_ids) 
      lstm_out, (hn, cn) = self.lstm(text_embeds, (h0, c0))
      text_features = hn.squeeze(0) 

      if mode == "vision+text":
          combined = torch.cat([pooled_features, text_features], dim=1)
      elif mode == "vision_only":
          combined = torch.cat([pooled_features, torch.zeros_like(text_features)], dim=1)
      elif mode == "text_only":
          combined = torch.cat([torch.zeros_like(pooled_features), text_features], dim=1)

      logits = self.classifier(combined)
      return logits

# -----------------------------------------------
# MODEL 2: ResNet-LSTM (Task 1 - Generation)
# -----------------------------------------------

class ResNetEncoder(nn.Module):
    
    def __init__(self, finetune=False):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_dim = 2048

        if not finetune:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images) # [B, 2048, 7, 7]
        pooled_features = self.gap(features).squeeze(-1).squeeze(-1) # [B, 2048]
        
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch_size, -1, self.encoder_dim) # [B, 49, 2048]
        return features, pooled_features

class BahdanauAttention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(decoder_dim, attention_dim)
        self.W_f = nn.Linear(encoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        h_t = self.W_h(hidden_state).unsqueeze(1) # [B, 1, attn_dim]
        f_i = self.W_f(features)                  # [B, 49, attn_dim]
        e = torch.tanh(h_t + f_i)                # [B, 49, attn_dim]
        alpha = F.softmax(self.V(e), dim=1)      # [B, 49, 1]
        context = (features * alpha).sum(dim=1)  # [B, enc_dim]
        return context, alpha.squeeze(-1)

class LSTMDecoder(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, lstm_hidden, encoder_dim, attention_dim, dropout):
        super(LSTMDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, lstm_hidden, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, lstm_hidden)
        self.init_h = nn.Linear(encoder_dim, lstm_hidden)
        self.init_c = nn.Linear(encoder_dim, lstm_hidden)
        self.fc_out = nn.Linear(lstm_hidden + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden_state(self, pooled_features):
        h0 = self.init_h(pooled_features)
        c0 = self.init_c(pooled_features)
        return h0, c0

    def forward(self, features, pooled_features, captions):
        batch_size = features.size(0)
        max_len = captions.size(1)
        
        embeddings = self.embed(captions)
        h, c = self.init_hidden_state(pooled_features)
        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(features.device)

        for t in range(max_len):
            word_embed = embeddings[:, t, :]
            h, c = self.lstm_cell(word_embed, (h, c))
            context, _ = self.attention(features, h)
            output = self.fc_out(self.dropout(torch.cat([h, context], dim=1)))
            predictions[:, t, :] = output
            
        return predictions

class ResNetLSTMGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, encoder_dim, dropout, 
                 pad_idx, sos_idx, eos_idx, attention_dim=256):
        super(ResNetLSTMGenerationModel, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = LSTMDecoder(
            vocab_size, embed_dim, lstm_hidden, encoder_dim, attention_dim, dropout
        )
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, images, captions, lengths=None):
        features, pooled = self.encoder(images)
        # Teacher forcing: feed ground-truth captions
        outputs = self.decoder(features, pooled, captions)
        return outputs

    @torch.no_grad()
    def generate(self, images, max_len=64, vocab=None):
        self.eval()
        batch_size = images.size(0)
        device = images.device
        
        features, pooled = self.encoder(images)
        h, c = self.decoder.init_hidden_state(pooled)
        
        word = torch.full((batch_size,), self.sos_idx, dtype=torch.long).to(device)
        generated_seqs_ids = []
        generated_texts = []

        for t in range(max_len):
            word_embed = self.decoder.embed(word)
            h, c = self.decoder.lstm_cell(word_embed, (h, c))
            context, _ = self.decoder.attention(features, h)
            output = self.decoder.fc_out(torch.cat([h, context], dim=1))
            
            predicted_word_idx = output.argmax(dim=1)
            generated_seqs_ids.append(predicted_word_idx.unsqueeze(1))
            word = predicted_word_idx

        # Decode
        generated_seqs_ids = torch.cat(generated_seqs_ids, dim=1).cpu().numpy()
        for ids in generated_seqs_ids:
            words = []
            for idx in ids:
                if idx == self.eos_idx:
                    break
                if idx not in [self.pad_idx, self.sos_idx]:
                    words.append(vocab.idx2word[idx])
            generated_texts.append(' '.join(words))
            
        return generated_texts
