import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import math

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, Subset

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Embeddings(nn.Module):


    def __init__(self, vocab_size, d_model, max_len = 128):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1)

    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):  
            for i in range(0, d_model, 2):  
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)  
        return pe

    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:, :embedding.size(1)]   
        embedding = self.dropout(embedding)
        return embedding
    
    
def create_masks(inputs, outputs_input, outputs_target):
    
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype = torch.uint8)
        return mask.unsqueeze(0) # 상삼각행렬 생성 -> 행과 열을 뒤 바꾸어 하삼각행렬로 바꿈. (밑에가 다 0)
    
    inputs_mask = inputs != 58100
    inputs_mask = inputs_mask.to(device)
    inputs_mask = inputs_mask.unsqueeze(1).unsqueeze(1) # 각  input에 대해서 상삼각행렬에 대응하도록 설정.
    
    outputs_input_mask = outputs_input != 58100
    outputs_input_mask = outputs_input_mask.unsqueeze(1) 
    outputs_input_mask = outputs_input_mask & subsequent_mask(outputs_input.size(-1)).type_as(outputs_input_mask.data)
    outputs_input_mask = outputs_input_mask.unsqueeze(1)
    # masking을 해줌으로서, 

    
    outputs_target_mask = outputs_target != 58100
    
    return inputs_mask, outputs_input_mask, outputs_target_mask
        
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key  = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        
        
        
        
        scores = torch.matmul(query, key.permute(0 ,1 ,3, 2)) / math.sqrt(query.size(-1))
        
        # print(query.shape, key.shape, value.shape, mask.shape, scores.shape)

        
        scores = scores.masked_fill(mask == 0, -1e9) # masking 된 것에 매우 작은 수 부여 -> softmax 계산시 -inf 로 계산되어짐.
        weights = F.softmax(scores, dim = -1) # attention score 계산
        context = torch.matmul(weights, value)  # attention value 계산
        context = context.permute(0,2,1,3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        
        interacted = self.concat(context)
        
        return interacted
        
        
class FeedForward(nn.Module):
    
    def __init__(self, d_model, middle_dim = 2048):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out        
    
    
class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, embeddings, mask):
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, embeddings, encoded, src_mask, target_mask):
        query = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query = self.layernorm(query + embeddings)
        interacted = self.dropout(self.src_multihead(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded
    
    
class Transformer(nn.Module):
    
    def __init__(self, d_model, heads, num_layers, vocab_size):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = Embeddings(self.vocab_size, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        self.logit = nn.Linear(d_model, self.vocab_size)
        
    def encode(self, src_words, src_mask):
        src_embeddings = self.embed(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
            
        return src_embeddings

    def decode(self, target_words, target_mask, src_embeddings, src_mask):
        tgt_embeddings = self.embed(target_words)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings

    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim = 2)
        return out