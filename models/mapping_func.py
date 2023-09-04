import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import *
from configs.config import *
from models.rnn_base import CustomDynaRNN, CustomMultiRNN

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Embedding(nn.Module):
    def __init__(self, input_size, hidden_dims, output_size, act_fn=None, layer_norm=False, device='cuda'):
        super(Embedding, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.output_size = output_size
        self.layer_norm = layer_norm
        self.act_fn = act_fn()
        self.device = device
        last_dim = self.input_size
        net = []
        for dim in self.hidden_dims:
            net.extend([
                nn.Linear(last_dim, dim), 
                nn.LayerNorm(dim) if self.layer_norm else nn.Identity(),
                act_fn()
            ])
            last_dim = dim
        net.append(nn.Linear(self.hidden_dims[-1], self.output_size))
        self.net = nn.Sequential(*net)
        # self.net = nn.ModuleList()
        # for dim in self.hidden_dims:
        #     self.net.append(nn.Linear(last_dim, dim))
        #     last_dim = dim
        # self.net.append(nn.Linear(self.hidden_dims[-1], self.output_size))

    def forward(self, inputs):
        # inputs = torch.FloatTensor(inputs) if type(inputs) == np.ndarray else inputs
        # inputs = inputs.to(self.device)
        ndims = inputs.ndim
        output = inputs
        if ndims > 2:            
            output = torch.reshape(output, [-1] + [output.shape[-1]])
        output = self.net(output)
        # for i in range(len(self.hidden_dims)):
        #     output = self.net[i](output)
        #     if self.layer_norm:
        #         output = one_dim_layer_normalization(output)
        #     output = self.act_fn(output)
        # output = self.net[-1](output)
        if ndims > 2:
            output = torch.reshape(output, shape(inputs)[:2] + [int(output.shape[-1])])
        return output

class MlpEncoder(nn.Module):
    def __init__(self, input_size, hidden_dims, act_fn, device='cuda'):
        super(MlpEncoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn()
        self.device = device
        last_dim = input_size
        self.net = nn.ModuleList()
        for dim in self.hidden_dims:
            self.net.append(nn.Linear(last_dim, dim))
            last_dim = dim
    
    def forward(self, source_input):
        encode_output = torch.FloatTensor(source_input) if type(source_input) == np.ndarray else source_input
        encode_output = encode_output.to(self.device)
        for i in range(len(self.hidden_dims)):
            encode_output = self.net[i](encode_output)
            encode_output = one_dim_layer_normalization(encode_output)
            encode_output = self.act_fn(encode_output)
        return encode_output

class Real2Sim(nn.Module):
    def __init__(self, input_size, rnn_hidden_dims, rnn_cell, seq_length, act_fn, ob_shape, action_shape, mlp_layer, output_hidden_dims, layer_norm, emb_dynamic, transition, transition_decoder, target_mapping, device='cuda'):
        super(Real2Sim, self).__init__()
        self.input_size = input_size
        self.rnn_hidden_dims = rnn_hidden_dims
        self.ob_shape = ob_shape
        self.emb_dynamic = emb_dynamic
        self.output_hidden_dims = output_hidden_dims
        self.target_mapping = target_mapping
        self.rnn_cell = rnn_cell
        self.mlp_layer = mlp_layer
        self.action_shape = action_shape
        self.seq_length = seq_length
        self.act_fn = act_fn
        self.layer_norm = layer_norm
        self.device = device

        self.mlp_layer = mlp_layer
        self.transition = transition
        self.transition_decoder = transition_decoder
        
        if self.emb_dynamic:
            self.rnn = CustomDynaRNN(input_size=self.input_size, rnn_cell=self.rnn_cell, hidden_dims=self.rnn_hidden_dims, ob_shape=self.ob_shape, transition=self.transition, transition_decoder=self.transition_decoder, device=self.device)
        else:
            self.rnn = CustomMultiRNN(input_size=self.input_size, rnn_cell=self.rnn_cell, hidden_dims=self.rnn_hidden_dims, device=self.device)

        last_dim = self.rnn_hidden_dims[-1]
        self.net = nn.ModuleList()
        for h in self.output_hidden_dims:    
            self.net.append(nn.Linear(last_dim, h))
            last_dim = h
        mu_std_input_size = self.output_hidden_dims[-1] if len(self.output_hidden_dims) != 0 else self.rnn_hidden_dims[-1]
        self.fc_mu = nn.Linear(mu_std_input_size, self.ob_shape)
        self.fc_std = nn.Linear(mu_std_input_size, self.ob_shape)

    def rnn_output_decode(self, inputs):
        if self.emb_dynamic:
            mu, std, hidden_state = inputs
        else:
            encode_output, hidden_state = inputs
            for i in range(len(self.output_hidden_dims)):
                encode_output = self.net[i](encode_output)
                if self.layer_norm:
                    encode_output = one_dim_layer_normalization(encode_output)
                encode_output = torch.tanh(encode_output)
            output = encode_output
            mu = self.fc_mu(output)
            log_std = self.fc_std(output)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std) + 1e-6
        return torch.distributions.Normal(loc=mu, scale=std), hidden_state

    def forward(self, ob_ac_embedding, ac, var_length, prev_hidden_states):
        # ob_ac_embedding, ac, var_length, initial_state = inputs # initial state should be the infer state from last observation: no need, pre_state is the start of the trajectory
        # ob_ac_embedding, ac, initial_state = ndarray2tensor([ob_ac_embedding, ac, initial_state], device=DEVICE)

        # if self.mlp_layer is not None:
        #     output = self.mlp_layer(ob_ac_embedding)
        #     last_hidden_state = torch.zeros((output.shape[0], self.ob_shape))
        #     all_hidden_state = torch.zeros(output.shape[0:2] + [self.ob_shape])
        #     output = (output, all_hidden_state)
        # else:
        #     if self.emb_dynamic: # embeded DM
        #         output, last_hidden_state = self.dynamic_rnn(ob_ac_embedding, ac, var_length, prev_hidden_states) # initial state should be the 
        #     else:
        #         output, last_hidden_state = self.multi_rnn(ob_ac_embedding, var_length)
        output, last_hidden_state = self.rnn(ob_ac_embedding, ac, var_length, prev_hidden_states)
        output, all_hidden_state = self.rnn_output_decode(output)
        return output, all_hidden_state, last_hidden_state
    

    # def get_batch_zero(self, batch_size):
    #     return torch.zeros((1, batch_size, 256+self.ob_shape)).to(self.device)


class Sim2Real(nn.Module):
    def __init__(self, real_input_size, real2sim_input_size, hidden_dims, rnn_hidden_dims, emb_dim, ob_shape, ac_shape, res_struc, act_fn=None, rnn_cell=None, layer_norm=False, real_ob_input=False, encoder=None, device='cuda'):
        super(Sim2Real, self).__init__()
        self.real_input_size = real_input_size
        self.real2sim_input_size = real2sim_input_size
        self.hidden_dims = hidden_dims
        self.rnn_hidden_dims = rnn_hidden_dims
        self.emb_dim = emb_dim
        self.act_fn = act_fn()
        self.res_struc = res_struc
        self.layer_norm = layer_norm
        self.rnn_cell = rnn_cell
        self.real_ob_input = real_ob_input
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.encoder = encoder
        self.device = device

        self.fc_real = nn.Linear(self.real_input_size, int(self.emb_dim/2)).to(self.device) # (11, 128)
        self.fc_real2sim = nn.Linear(self.real2sim_input_size, int(self.emb_dim/2)).to(self.device) # (14, 128)
        self.fc0 = nn.Linear(int(self.emb_dim/2), self.emb_dim).to(self.device)
        last_dim = self.emb_dim
        self.multi_rnn = CustomMultiRNN(input_size=last_dim, rnn_cell=self.rnn_cell, hidden_dims=self.rnn_hidden_dims, device=self.device)  # CHECK: 这里rnn是没有的
        # self.rnn = nn.ModuleList()
        # for h in self.rnn_hidden_dims:
        #     self.rnn.append(self.rnn_cell(last_dim, h, batch_first=True))
        #     last_dim = h
        if not isinstance(self.hidden_dims, list):
            self.hidden_dims = [self.hidden_dims]
        net = []
        for h in self.hidden_dims:
            net.extend([
                nn.Linear(last_dim, h), 
                nn.LayerNorm(h) if self.layer_norm else nn.Identity(),
                act_fn(), 
            ])
            last_dim = h
        self.net = nn.Sequential(*net)
        self.fc1 = nn.Linear(last_dim, self.emb_dim).to(self.device)


    def forward(self, ob_real2sim, ob_real, ac):  
        # ob_real2sim, ob_real, ac = inputs
        if not self.real_ob_input: # False
            ob_real = ob_real2sim                     # 为什么这里把图像换成了ob_real2sim?
        if self.res_struc == ResnetStructure.EMBEDDING_RAS:
            ob_real2sim = torch.cat([ob_real2sim, ac], dim=-1)
        elif self.res_struc == ResnetStructure.EMBEDDING_RS:
            ob_real2sim = ob_real2sim
        else:
            raise NotImplementedError
        # ob_real2sim = torch.FloatTensor(ob_real2sim) if type(ob_real2sim) == np.ndarray else ob_real2sim
        # ob_real2sim = ob_real2sim.to(self.device)
        real2sim_embedding = self.act_fn(self.fc_real2sim(ob_real2sim))
        real_embedding = self.act_fn(self.fc_real(ob_real))
        embedding_var = real2sim_embedding + real_embedding 
        output = self.act_fn(self.fc0(embedding_var))
        if self.rnn_hidden_dims:
            assert self.emb_dim == self.rnn_hidden_dims[0], self.emb_dim
            # for i in range(len(self.rnn_hidden_dims)):
            #     hidden_state = torch.zeros((1, output.shape[0], self.rnn_hidden_dims[i])).to(self.device)
            #     output, _ = self.rnn[i](output, hidden_state)
            self.var_length = self.get_variable_length(ac)
            output, last_hidden_state = self.multi_rnn(output, self.var_length)
            output, all_rnn_state = output
        
        output = self.net(output)
        output = self.fc1(output)
        output = torch.reshape(output, shape(embedding_var)[:-1]+[int(output.shape[-1])])
        return output

    def get_variable_length(self, data):
        used = torch.sign(torch.max(torch.abs(data), dim=2)[0])
        length = torch.sum(used, dim=1)
        length = length.int()
        return length
