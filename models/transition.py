import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.util import *

from UtilsRL.misc.decorator import profile

LOG_STD_MAX = 2
LOG_STD_MIN = -20
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import UtilsRL.exp as exp

def soft_clip(input, min_v, max_v):
    output = max_v - F.softplus(max_v - input)
    output = min_v + F.softplus(output - min_v)
    return output

class Transition(nn.Module):
    '''
    '''
    def __init__(self, transition_hidden_dims, transition_trainable, obs_min, obs_max, ob_shape, ac_shape, act_fn=nn.Tanh(), device=exp.device):
        super(Transition, self).__init__()
        self.act_fn = act_fn()
        self.transition_hidden_dims = transition_hidden_dims
        self.transition_trainable = transition_trainable
        self.ob_shape = ob_shape
        self.ac_shape = ac_shape
        self.device = device
        self.obs_min, self.obs_max = ndarray2tensor([obs_min, obs_max], device=exp.device)
        last_dim = self.ob_shape + self.ac_shape
        
        net = []
        for hidden_dim in self.transition_hidden_dims:
            net.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim
            net.append(self.act_fn)
        net.append(nn.Linear(self.transition_hidden_dims[-1], self.ob_shape))
        self.net = nn.Sequential(*net)

    def forward(self, transition_input):
        ob_sim, ac = tuple(transition_input)
        ac, ob_sim = ndarray2tensor([ac, ob_sim], device=exp.device)
        norm_obs_sim = ob_sim
        if not self.transition_trainable:
            with torch.no_grad():
                norm_obs_sim = torch.clamp(norm_obs_sim, min=self.obs_min, max=self.obs_max)
                transition_input = torch.cat([norm_obs_sim, ac], dim=-1)
                next_ob_sim = transition_input
                next_ob_sim = self.net(next_ob_sim)
                next_ob_sim = (torch.tanh(next_ob_sim) + 1.001) / 2.0 * (self.obs_max - self.obs_min) + self.obs_min
        else:
            transition_input = torch.cat([norm_obs_sim, ac], dim=-1)
            next_ob_sim = transition_input
            next_ob_sim = self.net(next_ob_sim)
            # for m in self.net:
                # next_ob_sim = m(next_ob_sim)
            next_ob_sim = (torch.tanh(next_ob_sim) + 1.001) / 2.0 * (self.obs_max - self.obs_min) + self.obs_min
        return next_ob_sim

    def load(self, weights):
        # weights = torch.load(path)
        self.load_state_dict(weights)

    def save(self, path):
        torch.save(self.state_dict(), path)


class TransitionDecoder(nn.Module):
    '''
    '''
    def __init__(self, ob_shape, input_dim, hidden_dims, obs_min, obs_max, device=exp.device):
        super(TransitionDecoder, self).__init__()
        self.ob_shape = ob_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.obs_min, self.obs_max = ndarray2tensor([obs_min, obs_max], device=exp.device)

        last_dim = self.input_dim
        net = []
        for hidden_dim in self.hidden_dims:
            net.extend([
                nn.Linear(last_dim, hidden_dim), 
                nn.LayerNorm(hidden_dim), 
                nn.Tanh()
            ])
        self.net = nn.Sequential(*net)   
        # self.net = nn.ModuleList()
        # for hidden_dim in self.hidden_dims:
            # self.net.append(nn.Linear(last_dim, hidden_dim))
            # last_dim = hidden_dim
        if len(self.hidden_dims) != 0:
            self.fc_mu = nn.Linear(self.hidden_dims[-1], self.ob_shape)
            self.fc_std = nn.Linear(self.hidden_dims[-1], self.ob_shape)
        else:
            self.fc_mu = nn.Linear(self.input_dim, self.ob_shape)
            self.fc_std = nn.Linear(self.input_dim, self.ob_shape)

    def forward(self, rnn_output, transition_predict, ob_real_emb, ac):
        # rnn_output, transition_predict, ob_real_emb, ac = tuple(source_input)
        decode = torch.cat([rnn_output, transition_predict, ob_real_emb, ac], dim=-1).to(self.device)
    
        # for i in range(len(self.hidden_dims)):
        #     decode = self.net[i](decode)
        #     decode = one_dim_layer_normalization(decode)
        #     decode = torch.tanh(decode)
        decode = self.net(decode)

        mu = self.fc_mu(decode)
        mu = soft_clip(mu, self.obs_min, self.obs_max)
        log_std = self.fc_std(decode)
        log_std = soft_clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) + 1e-6
        self.std = std
        return mu, std

class TransitionLearner(nn.Module):
    def __init__(self, transition, transition_target, ob_shape, ac_shape, batch_size, lr, 
                l2_loss, device=exp.device):
        super(TransitionLearner, self).__init__()
        self.transition = transition
        self.transition_target = transition_target
        self.lr = lr
        self.ob_shape = ob_shape
        self.l2_loss = l2_loss
        self.batch_size = batch_size
        self.ac_shape = ac_shape
        self.device = device
        self.optimizer = torch.optim.Adam(self.transition.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # @profile
    def update_transition(self, obs, acs, next_obs, lr=None):
        idx = np.arange(obs.shape[0])
        np.random.shuffle(idx)
        start_id = 0
        mse_loss_list, l2_loss_list, max_error_list = [], [], []
        idx = np.concatenate([idx, idx])
        if lr is None:
            lr = self.lr
        else: # dynamically assign the lr to each params group of optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        assert self.batch_size <= obs.shape[0], f"batchsize {self.batch_size}, obs.shape {obs.shape[0]}"

        while start_id + self.batch_size <= obs.shape[0]:
            sample_obs = obs[idx[start_id:start_id + self.batch_size]]
            sample_acs = acs[idx[start_id:start_id + self.batch_size]]
            sample_next_obs = next_obs[idx[start_id:start_id + self.batch_size]]
            sample_obs, sample_acs, sample_next_obs = ndarray2tensor([sample_obs, sample_acs, sample_next_obs], device=exp.device)
            start_id += self.batch_size
            ## MSE loss
            next_obs_pred = self.transition([sample_obs, sample_acs])
            mse_loss = self.criterion(next_obs_pred, sample_next_obs)
            max_error = torch.max(torch.square(sample_next_obs - next_obs_pred))
            l2reg_loss = l2_regularization(self.transition)
            loss_with_reg = mse_loss + l2reg_loss * self.l2_loss
            ## backward the loss
            self.optimizer.zero_grad()
            loss_with_reg.backward()
            self.optimizer.step()

            mse_loss_list.append(mse_loss.item())
            max_error_list.append(max_error.item())
            l2_loss_list.append(l2reg_loss.item())

        return {
            "model_mse_error": np.mean(mse_loss_list), 
            "model_max_error": np.mean(max_error_list), 
            "model_l2_reg": np.mean(l2_loss_list)
        
        }

    def predict(self, obs, acs):
        with torch.no_grad():
            next_obs_pred_target = self.transition_target([obs, acs])
        return next_obs_pred_target

    def copy_params_to_target(self):
        self.transition_target.load(self.transition.state_dict())

    def load(self, weights):
        transition_weights, transition_target_weights = weights
        self.transition.load(transition_weights)
        self.transition_target.load(transition_target_weights)
