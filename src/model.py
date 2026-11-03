import torch
from torch import nn
from transformers import BertModel
from .config import BASE_MODEL

class UserBERTClassifier(nn.Module):
    def __init__(self, model_name=BASE_MODEL, dropout=.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 2))
    def forward(self, input_ids, attention_mask, post_mask):
        b, n, length = input_ids.shape
        ids = input_ids.reshape(b*n, length)
        attn = attention_mask.reshape(b*n, length)
        tokens = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1).to(tokens.dtype)
        post_vec = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        post_vec = post_vec.reshape(b, n, -1)
        pm = post_mask.unsqueeze(-1).to(post_vec.dtype)
        user_vec = (post_vec * pm).sum(1) / pm.sum(1).clamp(min=1e-9)
        return self.classifier(user_vec)
