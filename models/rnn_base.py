import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LOG_STD_MAX = 2
LOG_STD_MIN = -20
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import UtilsRL.exp as exp



class CustomMultiRNN(nn.Module):
    def __init__(self, input_size, rnn_cell, hidden_dims, device='cuda'):
        super(CustomMultiRNN, self).__init__()
        self.input_size = input_size
        self.rnn_cell = rnn_cell
        self.hidden_dims = hidden_dims
        self.device = device

        self.rnn = nn.ModuleList()
        last_dim = self.input_size
        for dim in self.hidden_dims:
            self.rnn.append(self.rnn_cell(last_dim, dim, batch_first=True))
            last_dim = dim

    def forward(self, inputs, sequence_length):
        ob_ac_embedding = inputs
        # ob_ac_embedding = torch.FloatTensor(ob_ac_embedding) if type(ob_ac_embedding) == np.ndarray else ob_ac_embedding
        # ob_ac_embedding = ob_ac_embedding.to(self.device)
        sequence_length = sequence_length.to('cpu') if sequence_length.device != 'cpu' else sequence_length
        output = ob_ac_embedding # (batchsize, timesteps, ob_shape+ac_shape)
        output = pack_padded_sequence(output, sequence_length, batch_first=True, enforce_sorted=False)
        next_rnn_state = torch.zeros((1, ob_ac_embedding.shape[0], self.hidden_dims[0])).to(self.device)
        for i in range(len(self.hidden_dims)):
            output, next_rnn_state = self.rnn[i](output, next_rnn_state)
        output, _ = pad_packed_sequence(output, batch_first=True)
        all_hidden_states = output
        last_hidden_state = next_rnn_state
        return (output, all_hidden_states), last_hidden_state



class CustomDynaRNN(nn.Module):
    def __init__(self, input_size, rnn_cell, hidden_dims, ob_shape, transition, transition_decoder, device='cuda'):
        super(CustomDynaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.rnn_cell = rnn_cell
        self.transition = transition
        self.transition_decoder = transition_decoder
        self.ob_shape = ob_shape
        self.device = device
        self.hidden_dim = self.hidden_dims[-1]
        self.layer_num = len(self.hidden_dims)

        # self.rnn = nn.ModuleList()
        # last_dim = self.input_size
        # for dim in self.hidden_dims:
        #     self.rnn.append(self.rnn_cell(last_dim, dim, batch_first=True))
        #     last_dim = dim
        
        # NOTE: now we directly use rnn and assume that the hidden dims are the same
        if hidden_dims[0] != hidden_dims[1]:
            raise ValueError('CustomDynaRNN: The first and second hidden dimensions must be the same.')
        self.rnn = self.rnn_cell(self.input_size, hidden_dims[0], num_layers=len(self.hidden_dims), batch_first=True)    
    
    def forward(self, ob_real_emb, ac, sequence_length, prev_hidden_states):
        '''
        params:
            inputs : (batchsize, timesteps, emb_dim+ac_shape)
            sequence_length : [len] * batchsize
        returns:
            mus : (batchsize, timesteps, hidden_dims[-1])
            stds : (batchsize, timesteps, hiddem_dims[-1])
            all_hidden_states : (batchsize, timesteps, sum(hidden_dims)+emb_dim)
            last_hidden_state : (batchsize, 1, sum(hidden_dims)+emb_dim)
        '''
        batchsize, timesteps, emb_dim = list(ob_real_emb.shape)
        # ob_real_emb, ac = ndarray2tensor([ob_real_emb, ac], device=DEVICE)
        # record the sequence length of each data
        sequence_length = sequence_length.cpu().numpy()
        # lens = {}
        # for t in range(timesteps):
        #     lens[t] = [int(t<s) for s in sequence_length]

        mus, stds = [], []
        all_return_states = []
        
        # we reshape the prev_hidden_states here and reshape them back at the end
        assert isinstance(prev_hidden_states, torch.Tensor)
        rnn_states = torch.split(prev_hidden_states[..., :-self.ob_shape], self.hidden_dims, dim=-1)
        rnn_states = torch.stack(rnn_states, dim=0)
        ob_sim = prev_hidden_states[..., -self.ob_shape:].unsqueeze(1)

        # next_rnn_state = torch.zeros((1, batchsize, emb_dim+self.ob_shape)).to(self.device) # one hidden layer for each
        # update_idx = np.arange(batchsize)
        for t in range(timesteps):
            update_idx = np.where( sequence_length>t )[0]  # CHECK: why +1?
            if len(update_idx) == 0:
                mus.append(torch.zeros((batchsize, 1, self.ob_shape)).to(self.device))
                stds.append(torch.ones((batchsize, 1, self.ob_shape)).to(self.device))
                next_return_state = torch.zeros([batchsize, 256+self.ob_shape]).to(self.device)
                all_return_states.append(next_return_state)
                continue
            # state = next_rnn_state[..., :-self.ob_shape] # (1, batchsize, emb_dim)
            # ob_sim = torch.unsqueeze(next_rnn_state[0, :, -self.ob_shape:], dim=1) # (batchsize, 1, ob_shape)
            
            next_ob_sim = self.transition([ob_sim, ac[:, t:t+1, :]]) # (batchsize, 1, ob_shape)
            full_ob = torch.cat([ob_real_emb[:, t:t+1, :], next_ob_sim, ac[:, t:t+1, :]], dim=-1) # (batch_size, 1, emb_dim+ob_shape+ac_shape)
            # full_ob = pack_padded_sequence(full_ob, lens[t], batch_first=True, enforce_sorted=False)

            # next_rnn_state = state
            
            # start_idx = 0
            output = full_ob
            new_output, new_rnn_states = self.rnn(output, rnn_states)
            
            rnn_states = torch.zeros([self.layer_num, batchsize, self.hidden_dim]).to(self.device)
            rnn_states[:, update_idx, :] = new_rnn_states[:, update_idx, :]
            output = torch.zeros([batchsize, 1, self.hidden_dim]).to(self.device)
            output[update_idx, :, :] = new_output[update_idx, :, :]
            
            mu, std = self.transition_decoder(output, next_ob_sim, ob_real_emb[:, t:t+1, :], ac[:, t:t+1, :])
            next_ob_sim_distribution = torch.distributions.Normal(loc=mu, scale=std)
            ob_sim = next_ob_sim_distribution.rsample() 

            mus.append(mu)
            stds.append(std)
            next_return_state = torch.cat([rnn_states.permute(1,0,2).reshape(batchsize, -1), ob_sim.squeeze(dim=1)], dim=-1)
            all_return_states.append(next_return_state)
            
            # all_hidden_states.append()
            # for i in range(len(self.hidden_dims)):
            #     o, h = self.rnn[i](output.contiguous(), rnn_states[:, i, :])
            #     # output : (batch_size, 1, self.hidden_dims[i])
            #     o, h = self.rnn[i](output.contiguous(), next_rnn_state[..., start_idx:start_idx+self.hidden_dims[i]].contiguous())               
            #     output = torch.zeros_like(o).to(self.device)
            #     output[update_idx, ...], next_rnn_state[0, update_idx, start_idx:start_idx+self.hidden_dims[i]] = o[update_idx, ...], h[0, update_idx, :]
            #     start_idx += self.hidden_dims[i]            

            # output, _ = pad_packed_sequence(output, batch_first=True)
            # mu, std = self.transition_decoder((output, next_ob_sim, ob_real_emb[:, t:t+1, :], ac[:, t:t+1, :])) 
            # next_ob_sim_distribution = torch.distributions.Normal(loc=mu, scale=std) # ()
            # next_rnn_state = torch.cat([next_rnn_state[..., :emb_dim], torch.unsqueeze(torch.squeeze(next_ob_sim_distribution.sample(), dim=1), dim=0)], dim=-1)
            # mus.append(mu)
            # stds.append(std)
            # all_return_states.append(next_rnn_state.permute((1, 0, 2)))

        mus = torch.cat(mus, dim=1)
        stds = torch.cat(stds, dim=1)
        all_return_states = torch.stack(all_return_states, dim=1)
        last_return_state = next_return_state
        return (mus, stds, all_return_states), last_return_state
    