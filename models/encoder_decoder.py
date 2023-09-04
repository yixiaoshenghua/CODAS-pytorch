import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.util import *

class Encoder(nn.Module):
    def __init__(self, stack_imgs, device='cuda'):
        super(Encoder, self).__init__()
        self.stack_imgs = stack_imgs
        self.device = device
        self.act_fn = nn.ReLU()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, imgs):
        # imgs = torch.FloatTensor(imgs) if type(imgs) == np.ndarray else imgs
        # imgs = imgs.to(self.device)
        ndims = imgs.ndim
        if ndims == 5: # [batch, horizon, h, w, c]
            if imgs.shape[1] == 1 or self.stack_imgs == 1:
                stack_imgs = imgs
            else:
                # padding zeros
                def stack_idx(idx):
                    pre_pad_img = torch.zeros([imgs.shape[0], idx] + list(list(imgs.shape[2:])), dtype=imgs.dtype)
                    post_pad_img = torch.zeros([imgs.shape[0], self.stack_imgs - 1 - idx] + list(imgs.shape[2:]), dtype=imgs.dtype)
                    stacked_imgs = torch.cat([pre_pad_img, imgs, post_pad_img], axis=1)
                    return stacked_imgs
                idx_list = tuple(list(range(self.stack_imgs)))
                st_imgs = list(map(stack_idx, idx_list))
                stack_imgs = torch.cat(st_imgs, axis=-1)[:, :-1 * (self.stack_imgs - 1)]
            hidden = torch.reshape(stack_imgs, [-1] + list(stack_imgs.shape[2:]))   # (batch*horizon, h, w, c)
        elif ndims == 4:
            stack_imgs = imgs
            hidden = imgs
        else:
            raise NotImplementedError
        hidden = hidden.permute(0, 3, 1, 2)
        hidden = self.feature_extractor(hidden)

        assert list(hidden.shape[1:]) == [1024], hidden.shape
        if ndims == 5:
            hidden = torch.reshape(hidden, shape(stack_imgs)[:2] + 
                                    [np.prod(hidden.shape[1:])])
        return hidden

class LargeEncoder(nn.Module):
    '''
    '''
    def __init__(self, stack_imgs=0, device='cuda'):
        super(LargeEncoder, self).__init__()
        self.stack_imgs = stack_imgs
        self.device = device
        self.act_fn = nn.ReLU()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )


    def _init_layers(self, inputs):
        inputs = torch.FloatTensor(inputs) if type(inputs) == np.ndarray else inputs
        inputs = inputs.to(self.device)
        input_dim = list(inputs.shape[1:])
        num_features_after_cnn = np.prod(list(self.feature_extractor(torch.randn(1, *input_dim)).shape))
        self.fc = nn.Linear(num_features_after_cnn, 1024)

    def forward(self, imgs):
        imgs = torch.FloatTensor(imgs) if type(imgs) == np.ndarray else imgs
        imgs = imgs.to(self.device)
        ndims = imgs.ndim
        if ndims == 5: # [batch, horizon, h, w, c]
            if imgs.shape[1] == 1 or self.stack_imgs == 1:
                stack_imgs = imgs
            else:
                # padding zeros
                def stack_idx(idx):
                    pre_pad_img = torch.zeros([imgs.shape[0], idx] + list(imgs.shape[2:]), dtype=imgs.dtype)
                    post_pad_img = torch.zeros([imgs.shape[0], self.stack_imgs - 1 - idx] + list(imgs.shape[2:]), dtype=imgs.dtype)
                    stacked_imgs = torch.cat([pre_pad_img, imgs, post_pad_img], axis=1)
                    return stacked_imgs
                idx_list = tuple(list(range(self.stack_imgs)))
                st_imgs = list(map(stack_idx, idx_list))
                stack_imgs = torch.cat(st_imgs, axis=-1)[:, :-1 * (self.stack_imgs - 1)]
            hidden = torch.reshape(stack_imgs, [-1] + list(stack_imgs.shape[2:]))
        elif ndims == 4:
            stack_imgs = imgs
            hidden = imgs
        else:
            raise NotImplemented
        hidden = hidden.permute(0, 3, 1, 2)
        hidden = self.feature_extractor(hidden)
        hidden = self.act_fn(self.fc(hidden))

        assert hidden.shape[1:] == [1024], hidden.shape
        if ndims == 5:
            hidden = torch.reshape(hidden, shape(stack_imgs)[:2] + 
                                    [np.prod(hidden.shape[1:])])
        return hidden

class Decoder(nn.Module):
    '''
    '''
    def __init__(self, input_size, output_size, device='cuda'):
        super(Decoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.final_channel = output_size
        self.act_fn = nn.ReLU()
        self.linear_extractor = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        # input_dim = list(state.shape[1:])
        num_features_before_cnn = 2048 # np.prod(list(self.linear_extractor(torch.randn(1, *input_dim)).shape))
        self.deconv1 = nn.ConvTranspose2d(num_features_before_cnn, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, self.final_channel, kernel_size=6, stride=2)

    def forward(self, state, data_shape):
        # state, data_shape = source_input
        # state = torch.FloatTensor(state) if type(state) == np.ndarray else state
        # state = state.to(self.device)
        hidden = self.linear_extractor(state)
        hidden = torch.reshape(hidden, [-1, hidden.shape[-1], 1, 1])
        hidden = self.act_fn(self.deconv1(hidden))
        hidden = self.act_fn(self.deconv2(hidden))
        hidden = self.act_fn(self.deconv3(hidden))
        mean = self.deconv4(hidden)
        mean = mean.permute(0, 2, 3, 1)
        mean = torch.reshape(mean, shape(state)[:-1] + data_shape)
        return mean

class LargeDecoder(nn.Module):
    '''
    '''
    def __init__(self, device='cuda'):
        super(LargeDecoder, self).__init__()
        self.device = device

    def _init_layers(self, source_input):
        state, data_shape = source_input
        final_channel = data_shape[2]
        state = torch.FloatTensor(state) if type(state) == np.ndarray else state
        state = state.to(self.device)
        self.act_fn = nn.ReLU()
        self.linear_extractor = nn.Sequential(
            nn.Linear(state.shape[1], 1024),
            nn.Linear(1024, 2048),
        )
        input_dim = list(state.shape[1:])
        num_features_before_cnn = np.prod(list(self.linear_extractor(torch.randn(1, *input_dim)).shape))
        self.deconv1 = nn.ConvTranspose2d(num_features_before_cnn, 256, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(32, final_channel, kernel_size=6, stride=2)

    def forward(self, source_input):
        state, data_shape = source_input
        state = torch.FloatTensor(state) if type(state) == np.ndarray else state
        state = state.to(self.device)
        state = self.linear_extractor(state)
        hidden = torch.reshape(hidden, [-1, 1, 1, hidden.shape[-1]])
        hidden = self.act_fn(self.deconv1(hidden))
        hidden = self.act_fn(self.deconv2(hidden))
        hidden = self.act_fn(self.deconv3(hidden))
        mean = self.act_fn(self.deconv4(hidden))
        mean = mean.permute(0, 2, 3, 1)
        mean = torch.reshape(mean, shape(state)[:-1] + data_shape)
        return mean