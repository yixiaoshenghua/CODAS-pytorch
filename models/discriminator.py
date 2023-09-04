import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import *
from configs.config import *
from models.rnn_base import *
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import UtilsRL.exp as exp

class TrajDiscriminator(nn.Module):
    def __init__(self, input_size, hid_dims, emb_hid_dim, discre_struc, output_size, layer_norm, 
                rnn_hidden_dims, rnn_cell, device='cuda'):
        super(TrajDiscriminator, self).__init__()
        self.hid_dims = hid_dims
        self.discre_struc = discre_struc
        self.output_size = output_size
        self.device = device
        self.input_size = input_size
        self.emb_hid_dim = emb_hid_dim
        self.layer_norm = layer_norm
        self.act_fn = nn.ReLU()
        self.rnn_cell = rnn_cell
        self.rnn_hidden_dims = rnn_hidden_dims
        
        if self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            self.fc0 = nn.Linear(self.input_size, self.emb_hid_dim)
        else:
            raise NotImplementedError

        net = []
        last_dim = self.emb_hid_dim
        for dim in self.hid_dims:
            net.extend([
                nn.Linear(last_dim, dim), 
                nn.LayerNorm(dim) if self.layer_norm else nn.Identity(),
                nn.ReLU()
            ])
            last_dim = dim
        self.net = nn.Sequential(*net)
        self.multi_rnn = CustomMultiRNN(input_size=last_dim, rnn_cell=self.rnn_cell, hidden_dims=self.rnn_hidden_dims, device=self.device)
        # self.rnn = nn.ModuleList()
        # for h in self.rnn_hidden_dims:
        #     self.rnn.append(self.rnn_cell(last_dim, h))
        #     last_dim = h
        self.fc1 = nn.Linear(self.rnn_hidden_dims[-1], self.output_size)
    
    @property
    def hidden_state_size(self):
        return sum(self.rnn_hidden_dims)
    
    def forward(self, ob, ac):
        if self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            # ob, ac = source_input
            self.var_length = self.get_variable_length(ob) ## An int32/int64 vector sized [batch_size]
            # ob, ac = ndarray2tensor([ob, ac], device=DEVICE)
            x = torch.cat([ob, ac], dim=-1)
        else:
            raise NotImplementedError
        
        x = self.fc0(x)
        x = self.act_fn(x)
        x = self.net(x)
        
        output, last_hidden_state = self.multi_rnn(x, self.var_length)
        # for i in range(len(self.rnn)):
        #     hidden_state = torch.zeros([1, x.shape[1], self.rnn_hidden_dims[i]]).to(self.device)
        #     x, last_hidden_state = self.rnn[i](x, hidden_state)
        x = output[0]
        p_h3 = nn.Identity()(self.fc1(x))

        return p_h3

    def get_structure(self):
        return self.discre_struc


    def get_variable_length(self, data):
        used = torch.sign(torch.max(torch.abs(data), dim=2)[0])
        length = torch.sum(used, dim=1)
        length = length.int()
        return length

class StateDistributionDiscriminator(nn.Module):
    def __init__(self, input_size, hid_dims, emb_hid_dim, discre_struc, output_size, layer_norm, 
                device='cuda'):
        super(StateDistributionDiscriminator, self).__init__()
        self.hid_dims = hid_dims
        self.discre_struc = discre_struc
        self.output_size = output_size
        self.emb_hid_dim = emb_hid_dim
        self.layer_norm = layer_norm
        self.act_fn = nn.ReLU()
        self.input_size = input_size
        self.device = device

        self.fc0 = nn.Linear(self.input_size, self.emb_hid_dim)
        last_dim = self.emb_hid_dim
        self.net = nn.ModuleList()
        for dim in self.hid_dims:
            self.net.append(nn.Linear(last_dim, dim))
            last_dim = dim
        self.fc1 = nn.Linear(self.hid_dims[-1], self.output_size)

    def forward(self, source_input):
        if self.discre_struc == DiscriminatorStructure.OB:
            _input = source_input
            _input = torch.FloatTensor(_input).to(self.device)
        elif self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            ob, ac = source_input
            ob = torch.FloatTensor(ob).to(self.device)
            ac = torch.FloatTensor(ac).to(self.device)
            _input = torch.cat([ob, ac], dim=-1)
        x = self.fc0(_input)
        x = self.act_fn(x)
        for i in range(len(self.fcs)):
            x = self.net[i](x)
            if self.layer_norm:
                x = one_dim_layer_normalization(x)
            x = self.act_fn(x)
        p_h3 = self.fc1(x)

        return p_h3

    def get_structure(self):
        return self.discre_struc

class ImgDiscriminator(nn.Module):
    def __init__(self, input_size, hid_dims, emb_hid_dim, discre_struc, output_size, layer_norm, 
                device='cuda'):
        super(ImgDiscriminator, self).__init__()
        self.hidden_dim = hid_dims
        self.discre_struc = discre_struc
        self.output_size = output_size
        self.layer_norm = layer_norm
        self.input_size = input_size
        self.device = device

        self.conv1 = nn.Conv2d(self.input_size, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1024 + 3, 256)
        self.fc2 = nn.Linear(256, self.output_size)

    def forward(self, source_input):
        if self.discre_struc == DiscriminatorStructure.OB:
            imgs = source_input
        elif self.discre_struc == DiscriminatorStructure.OB_AC_CONCATE:
            imgs, ac = source_input
        else:
            raise NotImplementedError
        imgs = torch.FloatTensor(imgs)
        ndims = imgs.ndim
        if ndims == 5:
            hidden = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        elif ndims == 4:
            hidden = imgs
        else:
            raise NotImplementedError
        hidden = nn.ReLU(self.conv1(hidden))
        hidden = nn.ReLU(self.conv2(hidden))
        hidden = nn.ReLU(self.conv3(hidden))
        hidden = nn.ReLU(self.conv4(hidden))
        hidden = nn.Flatten(hidden)
        assert list(hidden.shape[1:]) == [1024], list(hidden.shape)
        if ndims == 5:
            hidden = torch.reshape(hidden, shape(hidden)[:2] + np.prod(list(hidden.shape[1:])))
        imgs_emb = hidden
        imgs_emb = torch.cat([imgs_emb, ac], dim=-1)
        x = nn.Tanh(self.fc1(imgs_emb))
        p_h3 = nn.Identity(self.fc2(x))
        
        return p_h3

    def get_structure(self):
        return self.discre_struc