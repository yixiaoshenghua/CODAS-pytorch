from operator import itemgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs.config import *
from utils.util import *
from UtilsRL.misc.decorator import profile
import time

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import UtilsRL.exp as exp

class VarSeq:
    def __init__(self, sequence_length: int,
                 rollout_step: int, img_shape,
                 batch_size: int,
                 lr_dis: int, lr_gen: int, l2_coeff: float, dis_l2_coeff: float,
                 total_timesteps: int, decay_ratio: float,
                 dis_test: bool,
                 embedding: nn.Module, sim2real_mapping: nn.Module, real2sim_mapping: nn.Module,
                 reconstruct_clip: float,
                 discriminator: nn.Module, obs_discriminator: nn.Module, grad_clip_norm: float,
                 emb_dynamic: bool, cycle_loss: bool,
                 encoder: nn.Module, init_first_state: bool,
                 decoder: nn.Module,  label_image_test: bool,
                 ob_shape, ac_shape, lambda_a: float, lambda_b: float,
                 minibatch_size: int, merge_d_train: bool,  stack_imgs:int, random_set_to_zero:bool, device=exp.device):

        self.merge_d_train = merge_d_train
        self.random_set_to_zero = random_set_to_zero
        self.minibatch_size = minibatch_size
        self.l2_coeff = l2_coeff
        self.dis_l2_coeff = dis_l2_coeff
        self.cycle_loss = cycle_loss
        self.reconstruct_clip = reconstruct_clip
        self.emb_dynamic = emb_dynamic
        self.sequence_length = sequence_length # 500
        self.init_first_state = init_first_state
        self.rollout_step = rollout_step # 25
        assert self.sequence_length % self.rollout_step == 0
        self.rollout_times = int(self.sequence_length / self.rollout_step) # 20
        self.grad_clip_norm = grad_clip_norm
        self.grad_clip = 5.0
        self.decay_ratio = decay_ratio
        self.lr_dis = lr_dis
        self.lr_gen = lr_gen
        self.embedding = embedding
        self.sim2real_mapping = sim2real_mapping
        self.real2sim_mapping = real2sim_mapping
        self.discriminator = discriminator
        self.obs_discriminator = obs_discriminator
        self.batch_size = batch_size
        self.encoder = encoder
        self.decoder = decoder
        self.label_image_test = label_image_test
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b
        self.total_timesteps = total_timesteps
        self.ob_shape = ob_shape
        self.dis_test = dis_test
        self.ac_shape = ac_shape
        self.stack_imgs = stack_imgs
        self.policy_loss = None
        self.batch_zero_state = None
        self.batch_zero_O = None
        self.batch_zero_A = None
        self.device = exp.device
        self.img_shape = img_shape
        self.img_shape_to_list = [self.img_shape[ImgShape.WIDTH], self.img_shape[ImgShape.HEIGHT], self.img_shape[ImgShape.CHANNEL]]

        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr_dis)
        self.dis_scheduler_poly_lr_decay = PolynomialLRDecay(self.discriminator_optimizer, max_decay_steps=self.total_timesteps, end_learning_rate=self.lr_dis * (1 - self.decay_ratio), power=1.0)
        self.generator_optimizer = torch.optim.RMSprop([{'params': self.encoder.parameters(), 'lr':self.lr_gen}, {'params':self.decoder.parameters(), 'lr':self.lr_gen}, {'params':self.sim2real_mapping.parameters(), 'lr':self.lr_gen}, {'params': self.real2sim_mapping.parameters(), 'lr': self.lr_gen}, {'params': self.embedding.parameters(), 'lr': self.lr_gen}])
        self.gen_scheduler_poly_lr_decay = PolynomialLRDecay(self.generator_optimizer, max_decay_steps=self.total_timesteps, end_learning_rate=self.lr_gen * (1 - self.decay_ratio), power=1.0)

    # @profile
    def infer_data(self, S_r, O_r, A_r, S_sim, A_sim, A_r_first=None, O_r_first=None, prev_hidden_states=None):
        if A_r_first is None:
            A_r_first = self.get_batch_zero_A(self.batch_size)
        if O_r_first is None:
            O_r_first = self.get_batch_zero_O(self.batch_size)
        if prev_hidden_states is None:
            prev_hidden_states = self.get_batch_zero_state(self.batch_size)
            
        S_r, O_r, A_r, S_sim, A_sim, A_r_first, O_r_first, prev_hidden_states = ndarray2tensor([S_r, O_r, A_r, S_sim, A_sim, A_r_first, O_r_first, prev_hidden_states], device=exp.device)
        
        if self.stack_imgs > 1:
            stack_O_r = self.stack_images(O_r)
        else:
            stack_O_r = O_r

        A_r_prev = torch.cat([torch.unsqueeze(A_r_first, dim=1), A_r[:, :-1]], dim=1)
        ## var_length, var_length_sim
        var_length = self.get_variable_length(S_r)
        var_length_sim = self.get_variable_length(S_sim)
        mask = self.get_variable_mask(S_r)
        mask_sim = self.get_variable_mask(S_sim)
        # self.sim_mask = self.get_variable_mask(S_sim)
        
        ## encode (s, a) pairs into embedding before inputing it into RNN
        encoded_O_r = self.encoder(stack_O_r) # visual encoder
        encoded_pair = self.embedding(torch.cat([encoded_O_r, A_r_prev], dim=2))

        ## real2sim mapping predict
        hat_S_r_distribution, all_r2s_hidden_state, r2s_hidden_state = self.real2sim_mapping(encoded_pair, A_r_prev, var_length, prev_hidden_states=prev_hidden_states)

        ## hat_S_r_mask, hat_O_r_mask
        hat_S_r = hat_S_r_distribution.rsample()
        
        # A minimal result dict which is used by following computing graphs
        ret_dict = {
            "all_hidden_state": all_r2s_hidden_state, 
            "var_length": var_length, 
            "var_length_sim": var_length_sim, 
            "hat_S_r": hat_S_r, 
            "mask": mask, 
            "mask_sim": mask_sim, 
        }
        if not self.cycle_loss:
            ret_dict["all_cycle_hidden_state"] = all_r2s_hidden_state
        # res_dict = {}
        # # res_dict['hat_S_r'] = hat_S_r
        # res_dict['all_hidden_state'] = all_r2s_hidden_state # [10,500,267]
        # res_dict['var_length'] = var_length
        # res_dict['var_length_sim'] = var_length_sim
        # res_dict['hat_S_r'] = hat_S_r * mask # [10,500,11]
        # if not self.cycle_loss:
        #     res_dict['all_cycle_hidden_state'] = all_r2s_hidden_state
        return ret_dict

    # @profile
    def train_discriminator(self, O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state):
        '''
        O_r_rollout: (b_r, 25, 64, 64, 3) b_r<200
        O_r_first: (b_r, 64, 64, 3)
        A_r_rollout: (b_r, 25, 3)
        A_r_first: (b_r, 3)
        A_sim_rollout: (b_sim, 25, 3)
        A_sim_first: (b_sim, 3)
        S_sim_rollout: (b_sim, 25, 11)
        S_r_rollout: (b_r, 25, 11)
        prev_hidden_state: (b_r, 267)
        prev_cycle_hidden_state: (b_r, 267)
        
        '''
        ## transform data type
        # O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state = ndarray2tensor([O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state], device=DEVICE)
        
        ## shuffle the real data
        group_traj_len = S_r_rollout.shape[0]
        idx = np.arange(0, group_traj_len)
        np.random.shuffle(idx)
        idx = np.concatenate([idx, idx])
        ## shuffle the sim data
        group_traj_len_sim = S_sim_rollout.shape[0]
        idx_sim = np.arange(0, group_traj_len_sim)
        np.random.shuffle(idx_sim)
        idx_sim = np.concatenate([idx_sim, idx_sim])

        if self.merge_d_train or self.minibatch_size == -1:
            minibatch_size = max(group_traj_len, group_traj_len_sim)
        else:
            minibatch_size = self.minibatch_size

        for i in range(int(np.ceil(group_traj_len/minibatch_size))):
            # prepare data
            start_id, end_id = i * minibatch_size, (i + 1) * minibatch_size
            S_r_ph = S_r_rollout[idx[start_id:end_id]]
            A_r_ph = A_r_rollout[idx[start_id:end_id]]
            A_r_first_ph = A_r_first[idx[start_id:end_id]]
            O_r_ph = O_r_rollout[idx[start_id:end_id]]
            O_r_first_ph = O_r_first[idx[start_id:end_id]]
            S_sim_ph = S_sim_rollout[idx_sim[start_id:end_id]]
            A_sim_ph = A_sim_rollout[idx_sim[start_id:end_id]]
            A_sim_first_ph = A_sim_first[idx_sim[start_id:end_id]]
            prev_hidden_state_ph = prev_hidden_state[idx[start_id:end_id]]
            # prev_cycle_hidden_state_ph = prev_cycle_hidden_state[idx[start_id:end_id]]
            
            # discriminator ops
            with torch.no_grad():
                res_infer_dict = self.infer_data(
                    S_r=S_r_ph, O_r=O_r_ph, A_r=A_r_ph, 
                    S_sim=S_sim_ph, A_sim=A_sim_ph, A_r_first=A_r_first_ph, 
                    O_r_first=O_r_first_ph, prev_hidden_states=prev_hidden_state_ph,
                )
            hat_S_r, mask, mask_sim = itemgetter("hat_S_r", "mask", "mask_sim")(res_infer_dict)
            
            dis_fake = self.discriminator(hat_S_r, A_r_ph)
            dis_real = self.discriminator(S_sim_ph, A_sim_ph)
            dis_fake_filter = mask_filter(dis_fake, mask[..., 0])
            dis_real_filter = mask_filter(dis_real, mask_sim[..., 0])
            
            generator_loss = F.binary_cross_entropy_with_logits(dis_fake_filter, torch.zeros_like(dis_fake_filter))
            expert_loss = F.binary_cross_entropy_with_logits(dis_real_filter, torch.ones_like(dis_real_filter))
            
            logits = torch.cat([dis_fake_filter, dis_real_filter], dim=0)
            entropy = torch.mean(logit_bernoulli_entropy(logits))
            entropy_loss = -0.001 * entropy
            
            minimax_loss = generator_loss + expert_loss + entropy_loss
            
            l2_reg_dis_loss = l2_regularization(self.discriminator)
            dis_loss = minimax_loss + l2_reg_dis_loss * self.dis_l2_coeff
            
            self.dis_scheduler_poly_lr_decay.step()
            self.discriminator_optimizer.zero_grad()
            dis_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_norm)  # CHECK discriminator gradient
            self.discriminator_optimizer.step()
            
            # self.adjust_allowed = np.expand_dims(adjust_allowed, axis=0)
            ## dis_loss GanLoss.MINIMAX
            # res_infer_dict = self.infer_data(S_r_ph, O_r_ph, A_r_ph, S_sim_ph, A_sim_ph, self.adjust_allowed)
            # mask = self.get_variable_mask(S_r_ph)
            # sim_mask = self.get_variable_mask(S_sim_ph)
            # hat_S_r = res_infer_dict['hat_S_r']
            # dis_fake = self.discriminator((hat_S_r, A_r_ph))
            # dis_real = self.discriminator((S_sim_ph, A_sim_ph))
            # dis_real_filter = mask_filter(dis_real, sim_mask[..., 0])
            # dis_fake_filter = mask_filter(dis_fake, mask[..., 0])
            # generator_loss = F.binary_cross_entropy_with_logits(dis_fake_filter, torch.zeros_like(dis_fake_filter).to(self.device))
            # expert_loss = F.binary_cross_entropy_with_logits(dis_real_filter, torch.ones_like(dis_real_filter).to(self.device))
            # logits = torch.cat([dis_fake_filter, dis_real_filter], dim=0)
            # entropy = torch.mean(logit_bernoulli_entropy(logits))
            # entropy_loss = -0.001 * entropy
            # minimax_loss = expert_loss + generator_loss + entropy_loss
            # l2_reg_dis_loss = l2_regularization(self.discriminator)
            # dis_loss = minimax_loss + l2_reg_dis_loss * self.dis_l2_coeff
            # # tune the learning rate
            # self.dis_scheduler_poly_lr_decay.step()
            # self.discriminator_optimizer.zero_grad()
            # dis_loss.backward()
            # self.discriminator_optimizer.step()

            ## dis_accuracy_real, dis_accuracy_fake
            dis_real_prob = torch.sigmoid(dis_real_filter)
            dis_fake_prob = torch.sigmoid(dis_fake_filter)
            # dis_real_accuracy = (dis_real_prob > 0.5).float()
            # dis_fake_accuracy = (dis_fake_prob < 0.5).float()
            
            res_dict = {
                # "dis_real_output": dis_real_filter.detach().cpu().mean().item(), 
                # "dis_fake_output": dis_fake_filter.detach().cpu().mean().item(), 
                "dis_real_prob": dis_real_prob.detach().cpu().mean().item(), 
                "dis_fake_prob": dis_fake_prob.detach().cpu().mean().item(), 
                # "dis_real_accuracy": dis_real_accuracy.detach().cpu().mean().item(),
                # "dis_fake_accuracy": dis_fake_accuracy.detach().cpu().mean().item(), 
                "generator_loss": generator_loss.detach().cpu().item(), 
                "expert_loss": expert_loss.detach().cpu().item(),
                # "entropy_loss": entropy_loss.detach().cpu().item(),
                "minimax_loss": minimax_loss.detach().cpu().item(), 
                "l2_reg_loss": l2_reg_dis_loss.detach().cpu().item(),
                "all_loss": dis_loss.detach().cpu().item(),
                "learning_rate": self.dis_scheduler_poly_lr_decay.get_lr()[0]
            }

        return res_dict

    # @profile
    def train_mapping(self, O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state):
        beg = time.time()
        ## transform data type
        O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state = ndarray2tensor([O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state], device=exp.device)
        ## shuffle the real data
        group_traj_len = S_r_rollout.shape[0]
        idx = np.arange(0, group_traj_len)
        np.random.shuffle(idx)
        idx = np.concatenate([idx, idx])
        ## shuffle the sim data
        group_traj_len_sim = S_sim_rollout.shape[0]
        idx_sim = np.arange(0, group_traj_len_sim)
        np.random.shuffle(idx_sim)
        idx_sim = np.concatenate([idx_sim, idx_sim])

        if self.minibatch_size == -1:
            minibatch_size = max(group_traj_len, group_traj_len_sim)
        else:
            minibatch_size = self.minibatch_size

        for i in range(int(np.ceil(group_traj_len/minibatch_size))):
            start_id, end_id = i * minibatch_size, (i + 1) * minibatch_size
            S_r_ph = S_r_rollout[idx[start_id:end_id]]
            A_r_ph = A_r_rollout[idx[start_id:end_id]]
            A_r_first_ph = A_r_first[idx[start_id:end_id]]
            O_r_ph = O_r_rollout[idx[start_id:end_id]]
            O_r_first_ph = O_r_first[idx[start_id:end_id]]
            S_sim_ph = S_sim_rollout[idx_sim[start_id:end_id]]
            A_sim_ph = A_sim_rollout[idx_sim[start_id:end_id]]
            A_sim_first_ph = A_sim_first[idx_sim[start_id:end_id]]
            prev_hidden_state_ph = prev_hidden_state[idx[start_id:end_id]]
            prev_cycle_hidden_state_ph = prev_cycle_hidden_state[idx[start_id:end_id]]
            
            res_infer_dict = self.infer_data(
                S_r=S_r_ph, O_r=O_r_ph, A_r=A_r_ph, 
                S_sim=S_sim_ph, A_sim=A_sim_ph, A_r_first=A_r_first_ph, 
                O_r_first=O_r_first_ph, prev_hidden_states=prev_hidden_state_ph,
            )
            hat_S_r, mask, mask_sim = itemgetter("hat_S_r", "mask", "mask_sim")(res_infer_dict)

            A_r_prev = torch.cat([torch.unsqueeze(A_r_first_ph, dim=1), A_r_ph[:, :-1]], dim=1)
            O_r_prev = O_r_ph
            O_r_prev = torch.cat([O_r_first_ph.unsqueeze(dim=1), O_r_prev[:, :-1]], dim=1)
            if self.stack_imgs > 1:
                stack_O_r_prev = self.stack_images(O_r_prev)
            else:
                stack_O_r_prev = O_r_prev
            encoded_O_r_prev = self.encoder(stack_O_r_prev)
            encoded_hat_O_r = self.sim2real_mapping(hat_S_r, encoded_O_r_prev, A_r_prev)
            hat_O_r = self.decoder(encoded_hat_O_r, self.img_shape_to_list)
            
            reconstruction_loss = F.binary_cross_entropy_with_logits(
               mask_filter(hat_O_r, mask[..., 0]), 
               mask_filter(O_r_ph, mask[..., 0])
            )
            dis_fake = self.discriminator(hat_S_r, A_r_ph)
            dis_fake_filter = mask_filter(dis_fake, mask[..., 0])
            generator_loss = - F.binary_cross_entropy_with_logits(dis_fake_filter, torch.zeros_like(dis_fake_filter))
            # logits_flat = torch.flatten(mask_filter(hat_O_r, mask[..., 0]), start_dim=1)
            # labels_flat = torch.flatten(mask_filter(O_r_ph, mask[..., 0]), start_dim=1)
            # real2sim2real_logprob = F.binary_cross_entropy_with_logits(logits_flat, labels_flat) 
            # real2sim2real_mse = F.mse_loss(logits_flat.detach(), labels_flat.detach()).cpu().item()
            # reconstruction_loss = torch.mean(real2sim2real_logprob)
            # mapping_likelihood = - reconstruction_loss
            
            # dis_fake = self.discriminator(hat_S_r, A_r_ph)
            # dis_fake_filter = mask_filter(dis_fake, mask[..., 0])
            # generator_loss = F.binary_cross_entropy_with_logits(dis_fake_filter, torch.zeros_like(dis_fake_filter).to(self.device))
            # generator_loss = - torch.mean(generator_loss)

            l2_reg_loss = 0
            l2_reg_loss += l2_regularization(self.sim2real_mapping)
            l2_reg_loss += l2_regularization(self.encoder)
            l2_reg_loss += l2_regularization(self.decoder)
            
            mapping_loss = self.l2_coeff * l2_reg_loss + \
                           self.lambda_a * generator_loss + \
                           self.lambda_b * reconstruction_loss
            self.gen_scheduler_poly_lr_decay.step()
            self.generator_optimizer.zero_grad()
            mapping_loss.backward()
            total_norm = nn.utils.clip_grad_norm_(list(self.real2sim_mapping.parameters())+
                                    list(self.sim2real_mapping.parameters())+
                                    list(self.encoder.parameters())+ 
                                    list(self.decoder.parameters()), max_norm=1000)
            if self.grad_clip_norm > 0:
                for k, p in self.real2sim_mapping.state_dict().items():
                    if "weight" in k:
                        nn.utils.clip_grad_value_(p, self.grad_clip_norm)
                for k, p in self.sim2real_mapping.state_dict().items():
                    if "weight" in k:
                        nn.utils.clip_grad_value_(p, self.grad_clip_norm)
                for k, p in self.encoder.state_dict().items():
                    if "weight" in k:
                        nn.utils.clip_grad_value_(p, self.grad_clip_norm)
                for k, p in self.decoder.state_dict().items():
                    if "weight" in k:
                        nn.utils.clip_grad_value_(p, self.grad_clip_norm)
            self.generator_optimizer.step()
            # res_infer_dict = self.infer_data(S_r_ph, O_r_ph, A_r_ph, S_sim_ph, A_sim_ph, self.adjust_allowed)
            # mask = self.get_variable_mask(S_r_ph)
            # hat_S_r = res_infer_dict['hat_S_r']
            # O_r_prev = O_r_ph
            # O_r_first_ph = self.get_batch_zero_O(O_r_ph.shape[0]).to(self.device)
            # O_r_prev = torch.cat([torch.unsqueeze(O_r_first_ph, dim=1), O_r_prev[:, :-1]], dim=1)

            # encoded_O_r_prev = self.encoder(stack_O_r_prev)
            # encoded_hat_O_r = self.sim2real_mapping([hat_S_r, encoded_O_r_prev, A_r_ph])
            # hat_O_r = self.decoder([encoded_hat_O_r, self.img_shape_to_list])

            # ## generator
            # logits_flat = torch.flatten(mask_filter(hat_O_r, mask[..., 0]), start_dim=1)
            # labels_flat = torch.flatten(mask_filter(O_r_ph, mask[..., 0]), start_dim=1)
            # real2sim2real_logprob = F.binary_cross_entropy_with_logits(logits_flat, labels_flat)
            # real2sim2real_mse = F.mse_loss(logits_flat.detach(), labels_flat.detach()).cpu().item()
            # reconstruct_loss = torch.mean(real2sim2real_logprob)
            # mapping_likelihood = reconstruct_loss
            # dis_fake = self.discriminator([hat_S_r, A_r_ph])
            # dis_fake_filter = mask_filter(dis_fake, mask[..., 0])
            # generator_loss = F.binary_cross_entropy_with_logits(dis_fake_filter, torch.zeros_like(dis_fake_filter).to(self.device))
            # generator_loss = -torch.mean(generator_loss)
            # ## l2 reg loss
            # l2_reg_loss = 0
            # l2_reg_loss += l2_regularization(self.sim2real_mapping)
            # l2_reg_loss += l2_regularization(self.encoder)
            # l2_reg_loss += l2_regularization(self.decoder)
            # ## mapping_loss
            # mapping_loss = self.lambda_a * generator_loss + self.lambda_b * mapping_likelihood + self.l2_coeff * l2_reg_loss
            # ## gen_learning_rate
            # self.gen_scheduler_poly_lr_decay.step()
            # self.generator_optimizer.zero_grad()
            # mapping_loss.backward()
            
            # ## clip grads
            # total_norm = nn.utils.clip_grad_norm_(list(self.real2sim_mapping.parameters())+
            #                         list(self.sim2real_mapping.parameters())+
            #                         list(self.encoder.parameters())+ 
            #                         list(self.decoder.parameters()), max_norm=1000)
            # if self.grad_clip_norm > 0:
            #     for k, p in self.real2sim_mapping.state_dict().items():
            #         if "weight" in k:
            #             nn.utils.clip_grad_value_(p, self.grad_clip_norm)
            #     for k, p in self.sim2real_mapping.state_dict().items():
            #         if "weight" in k:
            #             nn.utils.clip_grad_value_(p, self.grad_clip_norm)
            #     for k, p in self.encoder.state_dict().items():
            #         if "weight" in k:
            #             nn.utils.clip_grad_value_(p, self.grad_clip_norm)
            #     for k, p in self.decoder.state_dict().items():
            #         if "weight" in k:
            #             nn.utils.clip_grad_value_(p, self.grad_clip_norm)
            # self.generator_optimizer.step()
        
        return {
            "reconstruction_loss": reconstruction_loss.detach().cpu().item(), 
            "generator_loss": generator_loss.detach().cpu().item(),
            "l2_reg_loss": l2_reg_loss.detach().cpu().item(),
            "all_loss": mapping_loss.detach().cpu().item(),
            "learning_rate": self.gen_scheduler_poly_lr_decay.get_lr()[0], 
            # "O_reconstruction_mse": real2sim2real_mse,
        }

    # @profile
    def preprocess_data(self, S_r, O_r, A_r, S_sim, A_sim, var_length, var_length_sim, all_hidden_state, all_cycle_hidden_state):
        ## transform data type
        S_r, O_r, A_r, S_sim, A_sim, all_hidden_state, all_cycle_hidden_state = ndarray2tensor([S_r, O_r, A_r, S_sim, A_sim, all_hidden_state, all_cycle_hidden_state], device=exp.device)
        var_length = var_length.cpu().numpy()
        var_length_sim = var_length_sim.cpu().numpy()
        
        batch_length = []
        for i in range(self.rollout_times):
            batch_length.append(np.clip(var_length - i * self.rollout_step, 0, self.rollout_step))
        batch_length = np.concatenate(np.array(batch_length).T, axis=0)
        filter_idx = np.where(batch_length > 0)

        batch_length_sim = []
        for i in range(self.rollout_times):
            batch_length_sim.append(np.clip(var_length_sim - i * self.rollout_step, 0, self.rollout_step))
        batch_length_sim = np.concatenate(np.array(batch_length_sim).T, axis=0)
        filter_idx_sim = np.where(batch_length_sim > 0)

        all_hidden_state, S_r_rollout, A_r_rollout, O_r_rollout = map(self.data_reshape, (all_hidden_state, S_r, A_r, O_r))
        S_sim_rollout, A_sim_rollout = map(self.data_reshape, (S_sim, A_sim)) # (<200, 25, ...)
        mask_zero_idx = [i * self.rollout_times for i in range(self.batch_size)]

        def first_data_gen(zero_data, input_data):
            first_data = torch.cat([torch.unsqueeze(zero_data, dim=0), input_data[:-1, -1]], dim=0)
            first_data[mask_zero_idx] = zero_data
            return first_data

        batch_size = self.batch_size
        A_r_first = first_data_gen(self.get_batch_zero_A(batch_size)[0], A_r_rollout)
        A_sim_first = first_data_gen(self.get_batch_zero_A(batch_size)[0], A_sim_rollout)
        O_r_first = first_data_gen(self.get_batch_zero_O(batch_size)[0], O_r_rollout)
        prev_hidden_state = first_data_gen(self.get_batch_zero_state(batch_size)[0], all_hidden_state)
        if self.init_first_state:
            prev_hidden_state[mask_zero_idx][:, -1 * self.ob_shape:] = S_r[:, 0]
        # random set to zero data.
        if self.random_set_to_zero: # False
            A_r_first[:] = self.get_batch_zero_A(batch_size)[0]
            A_sim_first[:] = self.get_batch_zero_A(batch_size)[0]
            O_r_first[:] = self.get_batch_zero_O(batch_size)[0]

        def filter_zero_data(input_data, filter):
            return input_data[filter]

        prev_hidden_state, S_r_rollout, A_r_rollout, O_r_rollout, A_r_first, O_r_first = map(lambda x: filter_zero_data(x, filter_idx), (prev_hidden_state, S_r_rollout, A_r_rollout, O_r_rollout, A_r_first, O_r_first))

        S_sim_rollout, A_sim_rollout, A_sim_first = map(lambda x: filter_zero_data(x, filter_idx_sim), (S_sim_rollout, A_sim_rollout, A_sim_first))


        if not self.cycle_loss:
            prev_cycle_hidden_state = prev_hidden_state
        return O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state

    @torch.no_grad()
    def infer_step(self, img, prev_ac, prev_state, stoc_infer):
        stack_O_r = img[None, :]
        A_r_prev = prev_ac[None, :]
        r2s_prev_hidden_state = prev_state
        var_length = torch.LongTensor([1]).to(exp.device)
        
        stack_O_r, A_r_prev, r2s_prev_hidden_state = ndarray2tensor([stack_O_r, A_r_prev, r2s_prev_hidden_state], device=exp.device)
        
        encoded_O_r = self.encoder(stack_O_r)
        encoded_pair = self.embedding(torch.cat([encoded_O_r, A_r_prev], dim=2))
        hat_S_r_distribution, all_r2s_hidden_state, r2s_hidden_state = \
            self.real2sim_mapping(encoded_pair, A_r_prev, var_length, r2s_prev_hidden_state)
        if stoc_infer:
            hat_S_r = hat_S_r_distribution.sample()
        else:
            hat_S_r = hat_S_r_distribution.loc
        hat_S_r = hat_S_r[0]
        if self.emb_dynamic:
            r2s_hidden_state[:, -self.ob_shape:] = hat_S_r
        return hat_S_r.cpu().numpy(), r2s_hidden_state.cpu().numpy()
    
    def stack_images(self, imgs):
        # padding zeros
        def stack_idx(idx):
            pre_pad_img = torch.zeros([imgs.shape[0], idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
            post_pad_img = torch.zeros([imgs.shape[0], self.stack_imgs - 1 - idx] + imgs.shape[2:].as_list(), dtype=imgs.dtype)
            stacked_imgs = torch.cat([pre_pad_img, imgs, post_pad_img], axis=1)
            return stacked_imgs

        idx_list = tuple(list(range(self.stack_imgs)))
        st_imgs = list(map(stack_idx, idx_list))
        stack_imgs = torch.cat(st_imgs, axis=-1)[:, :-1 * (self.stack_imgs - 1)]
        return stack_imgs

    def get_batch_zero_A(self, batch_size):
        # if self.batch_zero_A is None:
        return torch.zeros([batch_size, self.ac_shape]).to(self.device)

    def get_batch_zero_O(self, batch_size):
        # if self.batch_zero_O is None:
        self.batch_zero_O = torch.zeros([batch_size] + self.img_shape_to_list).to(self.device)
        return self.batch_zero_O

    def get_variable_length(self, data):
        used = torch.sign(torch.max(torch.abs(data), dim=2)[0])
        length = torch.sum(used, dim=1)
        length = length.int()
        return length

    def get_variable_mask(self, data):
        return torch.unsqueeze(torch.sign(torch.max(torch.abs(data), dim=2)[0]), dim=-1)

    def data_reshape(self, seq_data):
        return torch.reshape(seq_data, [seq_data.shape[0] * self.rollout_times, self.rollout_step] + list(seq_data.shape[2:]))

    def get_batch_zero_state(self, batch_size):
        # if self.batch_zero_state is None:
        return torch.zeros((batch_size, 256+self.ob_shape)).to(self.device)
