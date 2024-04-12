import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm
from model import Transformer, Embeddings, MultiHeadAttention, create_masks, FeedForward, EncoderLayer, DecoderLayer
from training import AdamWarmup, LossWithLS, CustomDataset
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


if __name__ == '__main__':
    
    def train(train_loader, transformer, criterion, epoch):
        
        transformer.train()
        sum_loss = 0
        count = 0
        
        for i, (inputs, output) in enumerate(train_loader):
            
            samples = inputs.shape[0]
            
            inputs = inputs.to(device)
            output = output.to(device)
            
            output_in = output
            output_target = output
            
            inputs_mask, output_in_mask, output_target_mask = create_masks(inputs, output_in, output_target)
            
            
            out = transformer(inputs, inputs_mask, output_in, output_in_mask)
            
            loss = criterion(out, output_target, output_target_mask)
            transformer_optimizer.optimizer.zero_grad()
            loss.backward()
            transformer_optimizer.step()
            
            sum_loss += loss.item() * samples
            count += samples
            
            if i % 1 == 0:
                print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))
                
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dataset = load_dataset("bentrevett/multi30k")

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    dataset2 = dataset.map(lambda e: tokenizer(e['en'], padding='max_length', max_length= 128), batched=True)
    dataset3 = dataset.map(lambda e: tokenizer(e['de'], padding='max_length', max_length= 128), batched=True)

    train_en, train_ge = torch.tensor(dataset2['train']['input_ids']), torch.tensor(dataset3['train']['input_ids'])
    valid_en, valid_ge = torch.tensor(dataset2['validation']['input_ids']), torch.tensor(dataset3['validation']['input_ids'])
    test_en, test_ge = torch.tensor(dataset2['test']['input_ids']), torch.tensor(dataset3['test']['input_ids'])

    train_dataset = CustomDataset(train_en, train_ge)
    valid_dataset = CustomDataset(valid_en, valid_ge)
    test_dataset = CustomDataset(test_en, test_ge)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size = 64, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = True, drop_last = True)

    d_model = 128
    heads = 4
    num_layers = 2
    num_layers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    vocab_len = len(tokenizer.get_vocab())

    transformer = Transformer(d_model = d_model , heads = heads, num_layers = num_layers, vocab_size = vocab_len)
    transformer = transformer.to(device)
    adam_optimizer = torch.optim.Adam(transformer.parameters())
    transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = 4000, optimizer = adam_optimizer)
    criterion = LossWithLS(vocab_len, 0.1)
    
    for epoch in range(epochs):
        
        train(train_loader, transformer, criterion, epoch)
        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')
