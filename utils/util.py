import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import UtilsRL.exp as exp

def data_normalization(input, state_mean_std):
    norm_input = (input - state_mean_std[0]) / state_mean_std[1]
    if len(input.shape) == 3:
        norm_input[np.where(np.all(input == 0, axis=2))] = 0
    return norm_input

def data_denormalization(input, state_mean_std):
    # input = torch.FloatTensor(input) if type(input) == np.ndarray else input
    # input = input.to(DEVICE)
    input = input.cpu().numpy() if type(input) != np.ndarray else input
    norm_input = (input * state_mean_std[1]) + state_mean_std[0]
    if len(input.shape) == 3:
        norm_input[np.where(np.all(input == 0, axis=2))] = 0
    return norm_input

def one_dim_layer_normalization(input):
    layer_norm = nn.LayerNorm(list(input.shape[1:])).to(exp.device)
    return layer_norm(input)

def shape(tensor):
    return list(tensor.shape)

def l2_regularization(model):
    weights_list = [param.view(-1) for name, param in model.named_parameters() if name[-4:] != 'bias']
    l2reg_loss = torch.sum(torch.cat(weights_list) ** 2)
    return l2reg_loss

def get_output_size(model, input_dim, dims):
    # input_dim = list(inputs.shape[1:])
    shape = list(model(torch.randn(1, *input_dim).to(exp.device)).shape)
    output_size = 1
    for i in dims:
        output_size *= shape[i]
    return output_size

def ndarray2tensor(data: list, device=exp.device):
    for i in range(len(data)):
        data[i] = torch.FloatTensor(data[i]) if type(data[i]) == np.ndarray else data[i]
        data[i] = data[i].to(exp.device)
    return data


def mask_filter(data, mask):
    mask = mask.type(torch.BoolTensor)
    return data[mask]

def logit_bernoulli_entropy(logits):
    ent = (1.0 - torch.sigmoid(logits)) * logits - F.logsigmoid(logits)
    return ent

from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr