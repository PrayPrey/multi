
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdamWarmup:
    
    def __init__(self, model_size, warmup_steps, optimizer):
        
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0
    
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
    
    def step(self):
        
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.lr = lr
        self.optimizer.step()
        
        
class LossWithLS(nn.Module):
    
    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size

    def forward(self, prediction, target, mask):


        prediction = prediction.view(-1, prediction.size(-1))  
        target = target.contiguous().view(-1)   
        mask = mask.float()
        mask = mask.view(-1)    
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)   
        loss = (loss.sum(1) * mask).sum() / mask.sum()
        return loss
    
    

            
class CustomDataset(Dataset):
    
    def __init__(self, inputs, output):

        self.inputs = inputs
        self.output = output

    def __getitem__(self, idx):

        inputs = self.inputs[idx]
        output = self.output[idx]
        
        return inputs, output

    def __len__(self):
        return len(self.output)