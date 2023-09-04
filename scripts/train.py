import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import gym
from collections import deque

from configs.argments import *
from models.encoder_decoder import Encoder, Decoder, LargeEncoder, LargeDecoder
from models.discriminator import TrajDiscriminator, StateDistributionDiscriminator, ImgDiscriminator
from models.mapping_func import Real2Sim, Sim2Real, Embedding, MlpEncoder
from models.transition import Transition, TransitionLearner, TransitionDecoder
from models.variance_seq import VarSeq

from stable_baselines import PPO2
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from utils.env_wrapper import make_vec_env, GeneratorWrapper, is_dapg_env
from utils.functions import compute_adjusted_r2, compute_rmse
from utils.transition_helper import sample_next_batch_data, sample_sim_training_data, safe_one_step_transition, obs_acs_reshape
from utils.policy_wrapper import PolicyWrapper
from utils.rollout import Runner, MappingHolder
from utils.mujoco_dset import Mujoco_Dset
from utils.util import *
from utils.replay_buffer import TrajectoryBuffer
import mj_envs

import reset_able_mj_env


####################################################################################################################################
#                                                                                                                                  #
#                                                     define the parameters                                                        #
#                                                                                                                                  #
####################################################################################################################################
gym.make("Hopper-v4")
args = get_param()

# /////////////// set logger info ///////////////
from UtilsRL.exp import setup, select_free_cuda
from UtilsRL.logger import TensorboardLogger
from UtilsRL.monitor import Monitor
from UtilsRL.misc.decorator import profile
logger = TensorboardLogger(args.log_dir, args.exp_name)
# DEVICE = select_free_cuda()
DEVICE = torch.device("cuda:0")
setup({}, logger, DEVICE)
# ////////////////////////////////////////////////////

## TODO assume the env is not robot env
is_robot_env = False

act_fn = getattr(nn, args.act_fn)
resc_act_fn = getattr(nn, args.resc_act_fn)
dyn_act_fn = getattr(nn, args.dyn_act_fn)
rnn_cell = nn.GRU if args.rnn_cell == 'GRU' else nn.LSTM

norm_std_str = '' if args.norm_std_bound == 1 else 'std-{}'.format(args.norm_std_bound)
cpb_str = '' if args.clip_policy_bound else 'ncpb'
img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size, ImgShape.CHANNEL: 3}
OBS_BOUND = 150 if is_dapg_env(args.env_id) else 100
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_path = osp.join(DATA_ROOT, "saved_model")
logger.log_str(f"Loading and saving objects from {load_path}", type="WARNING")


####################################################################################################################################
#                                                                                                                                  #
#                                                define and reset the environments                                                 #
#                                                                                                                                  #
####################################################################################################################################

model_path = osp.join(load_path, "ppo_{}_{}_full.zip".format(args.env_id, args.policy_timestep))
env_path = osp.join(load_path, "{}_full".format(args.env_id))
if np.abs(args.dual_policy_noise_std - 0.0) > 1e-5:
    assert False    #  debug
    real_expert_path = osp.join(DATA_ROOT, 'dual_{:.02f}_ppo_{}_{}_{}_deter_False_uint8.npz'.
                            format(args.dual_policy_noise_std, args.env_id, args.policy_timestep,
                                    args.collect_trajs))
    sim_expert_path = osp.join(DATA_ROOT, 'ppo_{}_{}_full_{}_deter_False_uint8_full.npz'.
                            format(args.env_id, args.policy_timestep, args.collect_trajs))
else:
    real_expert_path = sim_expert_path = osp.join(DATA_ROOT, 'ppo_{}_{}_full_{}_deter_False_uint8_full.npz'.
                            format(args.env_id, args.policy_timestep, args.collect_trajs))

env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=args.stoc_init_range)
env = VecNormalize.load(env_path, env)
env.training = False
env.norm_reward = False

real_world_env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=1.0, stoc_init_range=0.005)
real_world_env = VecNormalize.load(env_path, real_world_env)
real_world_env.training = False
real_world_env.norm_reward = False
model = PPO2.load(model_path)

real_world_env = GeneratorWrapper(real_world_env, use_image_noise=args.use_noise_env)
env = GeneratorWrapper(env)

dynamics_model_path = osp.join(DATA_ROOT, f'ppo_{args.env_id}_{args.policy_timestep}_{COLLECT_TRAJ}_network_weights-full'
                                            f'-{args.dynamic_param}-ca-{args.clip_acs}-'
                                            f'dn-{args.data_normalize}{args.minmax_normalize}{norm_std_str}{cpb_str}')
dynamics_model_param_path = osp.join(DATA_ROOT, f'ppo_{args.env_id}_{args.policy_timestep}_{COLLECT_TRAJ}_'
                                                f'network_weights_param-full-{ args.dynamic_param}-ca-{args.clip_acs}-'
                                                f'dn-{args.data_normalize}{norm_std_str}{cpb_str}')

if args.minmax_normalize:
    assert False     # debug
    dynamics_model_path += '-mn'
    dynamics_model_param_path += '-mm'

dynamics_model_path += '.npy'
dynamics_model_param_path += '.pkl'

logger.log_str(f"Load policy from {model_path}")
logger.log_str(f"Load real-world env and sim-env from {env_path}")
logger.log_str(f"Load real expert dataset from {real_expert_path}")
logger.log_str(f"Load sim expert dataset from {sim_expert_path}")
logger.log_str(f"Load/Save dynamics from {dynamics_model_path}")
logger.log_str(f"Load/Save dynamics parameter from {dynamics_model_param_path}")

runner = Runner(simulator_env=env, real_world_env=real_world_env, sim_policy=model, real_policy=model,
                max_horizon=args.max_sequence, img_shape=img_shape, clip_acs=args.clip_acs, exact_consist=args.exact_consist)

env.reset()
real_world_env.reset()

# compute expert reward
rews = []
for i in range(args.expert_perf_eval_times):
    ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False, run_in_realworld=True)
    while ret_dict[runner.TRAJ_LEN] == 0:
        ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=False, run_in_realworld=True)
    rews.append(ret_dict[runner.TOTAL_REW])
expert_reward = np.mean(rews)
logger.log_str("Expert reward: {}".format(expert_reward), type="WARNING")

####################################################################################################################################
#                                                                                                                                  #
#                                                      collect the dataset                                                         #
#                                                                                                                                  #
####################################################################################################################################

expert_dataset = Mujoco_Dset(sim_data=False, expert_path=real_expert_path, traj_limitation=args.traj_limitation,
                                 use_trajectory=True, max_sequence=args.max_sequence, env=env,
                                 data_used_fraction=args.data_used_fraction, clip_action=args.clip_acs,
                                 filter_traj=args.filter_traj, npmap_replace=args.npmap_replace)
sim_training_dataset = Mujoco_Dset(sim_data=True, expert_path=sim_expert_path, traj_limitation=-1,
                                use_trajectory=True, max_sequence=args.max_sequence, env=env,
                                data_used_fraction=1.0, clip_action=args.clip_acs, filter_traj=False,
                                    npmap_replace=args.npmap_replace)
expert_dataset.obs_std[expert_dataset.obs_std == 0] = 1
sim_training_dataset.obs_std[sim_training_dataset.obs_std < args.norm_std_bound] = 1

state_mean_std = [sim_training_dataset.obs_mean, sim_training_dataset.obs_std]
if not args.data_normalize:
    state_mean_std[0] = np.zeros(state_mean_std[0].shape)
    state_mean_std[1] = np.ones(state_mean_std[1].shape)

if args.minmax_normalize:
    state_mean_std[0] = sim_training_dataset.obs_min
    state_mean_std[1] = sim_training_dataset.obs_max - sim_training_dataset.obs_min
    state_mean_std[0][state_mean_std[1] == 0] = 0
    state_mean_std[1][state_mean_std[1] == 0] = 1
# state_mean_std = torch.FloatTensor(state_mean_std).to(DEVICE)      # CHECK: why commenting this ?

env.reset()
real_world_env.reset()

####################################################################################################################################
#                                                                                                                                  #
#                                                   define the normalize range                                                     #
#                                                                                                                                  #
####################################################################################################################################

if args.clip_policy_bound:
    norm_min = data_normalization(np.clip(sim_training_dataset.obs_min, -1 * OBS_BOUND, OBS_BOUND), state_mean_std)
    norm_max = data_normalization(np.clip(sim_training_dataset.obs_max, -1 * OBS_BOUND, OBS_BOUND), state_mean_std)
else:
    norm_min = data_normalization(sim_training_dataset.obs_min, state_mean_std)
    norm_max = data_normalization(sim_training_dataset.obs_max, state_mean_std)

norm_range = norm_max - norm_min

epsilon_expanded = 0.05
update_dynamics_range_min = norm_min - epsilon_expanded * norm_range
update_dynamics_range_max = norm_max + epsilon_expanded * norm_range
update_dynamics_range_min_trans_learn = norm_min - (epsilon_expanded - 1e-3) * norm_range
update_dynamics_range_max_trans_learn = norm_max + (epsilon_expanded - 1e-3) * norm_range

####################################################################################################################################
#                                                                                                                                  #
#                                                      initialize the models                                                       #
#                                                                                                                                  #
####################################################################################################################################

# define the transition moodels
if args.emb_dynamic:
    logger.log_str("Initializing embedding dynamics model ...")
    transition = Transition(transition_hidden_dims=args.dyn_hid_dims, transition_trainable=True,
                                        ob_shape=env.state_space.shape[0], ac_shape=env.action_space.shape[0], act_fn=dyn_act_fn,  
                                        obs_min=update_dynamics_range_min,
                                        obs_max=update_dynamics_range_max, device=DEVICE).to(DEVICE)
    transition_target = Transition(transition_hidden_dims=args.dyn_hid_dims, transition_trainable=False,
                                        ob_shape=env.state_space.shape[0], ac_shape=env.action_space.shape[0], act_fn=dyn_act_fn,  
                                        obs_min=update_dynamics_range_min,
                                        obs_max=update_dynamics_range_max, device=DEVICE).to(DEVICE)
    transition_learner = TransitionLearner(transition=transition, transition_target=transition_target, 
                                            ob_shape=env.state_space.shape[0], ac_shape=env.action_space.shape[0], 
                                            lr=args.lr_dyn, batch_size=args.dyn_batch_size, l2_loss=args.dyn_l2_loss, device=DEVICE).to(DEVICE)
    transition_decoder_input_size = args.r2s_rnn_hid_dims[-1] + env.state_space.shape[0] + 256 + env.action_space.shape[0]
    transition_decoder = TransitionDecoder(ob_shape=env.state_space.shape[0], input_dim=transition_decoder_input_size,
                                               hidden_dims=args.r2s_output_hid_dims,
                                               obs_min=update_dynamics_range_min,
                                               obs_max=update_dynamics_range_max, device=DEVICE).to(DEVICE)
else:
    transition = None
    transition_learner = None
    transition_decoder = None

# define the discriminator
if args.gan_loss == GanLoss.MINIMAX:
    if args.traj_dis: ## judge whether the discriminator is applied on trajectory or statedistribution
        logger.log_str("Initializing Trajectory Discriminator")
        discriminator = TrajDiscriminator(input_size=env.state_space.shape[0]+env.action_space.shape[0], hid_dims=args.disc_hid_dims, emb_hid_dim=args.disc_emb_hid_dim, output_size=1, discre_struc=args.dis_struc, layer_norm=False, rnn_cell=rnn_cell, rnn_hidden_dims=[128], device=DEVICE).to(DEVICE)
    else:
        logger.log_str("Initializing State Distribution Discriminator")
        state_dis_input_size = env.state_space.shape[0] if args.dis_struc == DiscriminatorStructure.OB else env.state_space.shape[0] + env.action_space.shape[0]
        discriminator = StateDistributionDiscriminator(input_size=state_dis_input_size, hid_dims=args.disc_hid_dims, emb_hid_dim=args.disc_emb_hid_dim, output_size=1, discre_struc=args.dis_struc, layer_norm=False, rnn_cell=rnn_cell, rnn_hidden_dims=[128]).to(DEVICE)
elif args.gan_loss == GanLoss.WGAN:
    logger.log_str("Initializing WGAN State Distrbution Discriminator")
    state_dis_input_size = env.state_space.shape[0] if args.dis_struc == DiscriminatorStructure.OB else env.state_space.shape[0] + env.action_space.shape[0]
    discriminator = StateDistributionDiscriminator(input_size=state_dis_input_size, hid_dims=args.disc_hid_dims, emb_hid_dim=args.disc_emb_hid_dim, output_size=1, discre_struc=args.dis_struc, layer_norm=False, act_fn=nn.LeakyReLU).to(DEVICE)
else:
    raise NotImplementedError

# for cycle loss
img_discriminator = ImgDiscriminator(input_size=img_shape[ImgShape.HEIGHT],hid_dims=args.disc_img_hid_dims, emb_hid_dim=args.disc_emb_hid_dim, output_size=1, discre_struc=args.dis_struc, layer_norm=args.layer_norm).to(DEVICE)

if args.mlp: # default False
    mlp = MlpEncoder(input_size=None, hidden_dims=args.dyn_hid_dims, act_fn=nn.Tanh).to(DEVICE)
else:
    mlp = None

# define the mapping function
real2sim_input_size = 256 + env.state_space.shape[0] + env.action_space.shape[0]  if args.emb_dynamic else 256
real2sim = Real2Sim(input_size=real2sim_input_size, rnn_hidden_dims=args.r2s_rnn_hid_dims, rnn_cell=rnn_cell, seq_length=args.max_sequence, act_fn=act_fn, ob_shape=env.state_space.shape[0], action_shape=env.action_space.shape[0], mlp_layer=mlp, output_hidden_dims=args.r2s_output_hid_dims, layer_norm=args.layer_norm, emb_dynamic=args.emb_dynamic, transition=transition, transition_decoder=transition_decoder, target_mapping=False, device=DEVICE).to(DEVICE)

sim2real_real_input_size = 256 if args.real_ob_input else env.state_space.shape[0]
sim2real_real2sim_input_size = env.state_space.shape[0] + env.action_space.shape[0] if args.res_struc == ResnetStructure.EMBEDDING_RAS else env.state_space.shape[0]
sim2real = Sim2Real(real_input_size=sim2real_real_input_size, real2sim_input_size=sim2real_real2sim_input_size, hidden_dims=args.s2r_emb_dim, rnn_hidden_dims=args.s2r_rnn_hid_dims, rnn_cell=rnn_cell, emb_dim=args.s2r_emb_dim, ob_shape=env.state_space.shape[0], ac_shape=env.action_space.shape[0], layer_norm=args.layer_norm, res_struc=args.res_struc, act_fn=resc_act_fn, real_ob_input=args.real_ob_input, device=DEVICE).to(DEVICE)

if args.image_size == 64:
    encoder = Encoder(stack_imgs=args.stack_imgs, device=DEVICE).to(DEVICE)
    decoder = Decoder(input_size=256, output_size=img_shape[ImgShape.CHANNEL], device=DEVICE).to(DEVICE)
else:
    assert args.image_size == 128
    encoder = LargeEncoder(stack_imgs=args.stack_imgs).to(DEVICE)
    decoder = LargeDecoder().to(DEVICE)

obs_output_size = get_output_size(encoder, [args.max_sequence]+list(img_shape.values()), dims=[-1])
embedding = Embedding(input_size=obs_output_size+env.action_space.shape[0], hidden_dims=args.emb_hid_dims, output_size=args.emb_output_size, act_fn = act_fn, layer_norm=args.layer_norm, device=DEVICE).to(DEVICE)

## define the variance sequence model
logger.log_str("Initializing Variational Sequential Model")
var_seq = VarSeq(sequence_length=args.max_sequence, img_shape=img_shape,
                     embedding=embedding, real2sim_mapping=real2sim, sim2real_mapping=sim2real,
                     discriminator=discriminator, obs_discriminator=img_discriminator,
                     encoder=encoder, decoder=decoder,
                     batch_size=args.trajectory_batch,
                     lambda_a=args.lambda_a, lambda_b=args.lambda_b,
                     ac_shape=env.action_space.shape[0], ob_shape=env.state_space.shape[0],
                     lr_dis=args.lr_dis, lr_gen=args.lr_gen,
                     total_timesteps=args.total_timesteps, decay_ratio=args.decay_ratio,
                     grad_clip_norm=args.grad_clip_norm,
                     dis_test=args.dis_test, label_image_test=args.label_image_test,
                     reconstruct_clip=args.reconstruct_clip,
                     emb_dynamic=args.emb_dynamic, rollout_step=args.rollout_step,
                     cycle_loss=args.cycle_loss, minibatch_size=args.minibatch_size, merge_d_train=args.merge_d_train,
                     stack_imgs=args.stack_imgs, random_set_to_zero=args.random_set_to_zero,
                     init_first_state=args.init_first_state, l2_coeff=args.mapping_l2_loss,
                     dis_l2_coeff=args.dis_l2_loss, device=DEVICE)

####################################################################################################################################
#                                                                                                                                  #
#                                                   pretrain the dynamic model                                                     #
#                                                                                                                                  #
####################################################################################################################################

pool_size = int(args.pool_size if args.pool_size > 0 else args.trajectory_batch * args.data_reused_times * 1000)
trajecory_buffer = TrajectoryBuffer(pool_size, has_img=False)
adjust_allowed = args.adjust_allowed    # CHECK: adjust allowed 看起来没什么作用，确认一下

max_error_list = deque(maxlen=50)
max_error_list.append(np.inf)

if args.emb_dynamic and args.load_data == '':
    if osp.exists(dynamics_model_path) and osp.exists(dynamics_model_param_path) and not args.retrain_dynamics:
        logger.log_str(f"Loading embedded dynamics model from {dynamics_model_path} and {dynamics_model_param_path}")
        weights = torch.load(dynamics_model_path, map_location="cpu")
        transition_learner.transition.load(weights)
        hp_dict = pickle.load(open(dynamics_model_param_path, 'rb'))
        transition_learner.lr = hp_dict['lr']
    else:
        logger.log_str("Training embedded dynamics model from scratch", type="WARNING")
        break_counter = 0
        inc_batch_counter = 0
        counter = 0
        while True:
            obs_real, _, acs_real, lengths = sample_sim_training_data(args, sim_training_dataset, OBS_BOUND, state_mean_std, traj_batch_size=100)
            obs_train, acs_train, obs_next_train = obs_acs_reshape(obs_real, acs_real)

            train_res = transition_learner.update_transition(obs_train, acs_train, obs_next_train, lr=args.dyn_lr_pretrain)
            if counter % 50 == 0:
                transition_learner.copy_params_to_target()
                # use logger to record training loss
                train_res.update({
                    "inc_batch_counter": inc_batch_counter, 
                    "break_counter": break_counter
                })
                logger.log_scalars("Embedded DM Pretrain", train_res, step=counter)
                logger.log_dict("Embedded DM Pretrain", train_res)

            inc_batch_counter = inc_batch_counter + 1 if np.min(max_error_list) <= train_res["model_max_error"] else 0
            max_error_list.append(train_res["model_max_error"])

            if inc_batch_counter >= 200 and counter > 100000: # store the weights
                inc_batch_counter = 0
                transition_learner.transition.save(dynamics_model_path)
                with open(dynamics_model_param_path, 'wb') as f:
                    pickle.dump({"lr": transition_learner.lr}, file=f)
                break
            
            if counter > 300000:
                break
            
            if counter % 1000 == 0: # store the weights
                transition_learner.transition.save(dynamics_model_path)
                with open(dynamics_model_param_path, 'wb') as f:
                    pickle.dump({"lr": transition_learner.lr}, file=f)

            break_counter += 1 if train_res["model_max_error"] < (adjust_allowed * 0.8)**2 else 0
            if break_counter >= 20 and counter > 100000:
                break
            counter += 1
            # print("Counter {}\tmse_loss {:.4f}\tmax_error {:.4f}\tl2_reg {:.4f}".format(counter, mse_loss, max_error, l2_reg))

####################################################################################################################################
#                                                                                                                                  #
#                                        train the discriminator and mapping function                                              #
#                                                                                                                                  #
####################################################################################################################################

total_timesteps = args.total_timesteps
logger.log_str(f"Start training, total timesteps = {total_timesteps}")
torch.cuda.empty_cache()

start_epoch = 0
mapping_train_epoch = args.mapping_train_epoch
too_strong_discriminator = False
for it in Monitor("var_seq").listen(range(start_epoch, total_timesteps)):
    ## sample a batch of target domain trajectories real traj = {(o0,a0,o1,a1,...)} from real dataset
    obs_real, imgs_real, acs_real, lengths = sample_next_batch_data(args, runner, expert_dataset, trajecory_buffer, OBS_BOUND, state_mean_std, LossType.VAE, it)
    obs_sim, _, acs_sim, lengths = sample_next_batch_data(args, runner, expert_dataset, trajecory_buffer, OBS_BOUND, state_mean_std, LossType.GAN, it)

    ## infer the corresponding state trajectories sim traj = {(s^hat_1, ..., s^hat_T)} via the mapping function f
    with torch.no_grad():
        res_infer_dict = var_seq.infer_data(S_r=obs_real, O_r=imgs_real, A_r=acs_real, S_sim=obs_sim, A_sim=acs_sim, prev_hidden_states=None)

    var_length, var_length_sim = res_infer_dict['var_length'], res_infer_dict['var_length_sim']
    all_hidden_state, all_cycle_hidden_state = res_infer_dict['all_hidden_state'], res_infer_dict['all_cycle_hidden_state']
    ob_real2sim = res_infer_dict["hat_S_r"] * res_infer_dict["mask"]

    # no need to perform full_state_to_state here because no obs are fed into networks
    ## rollout one step with the oracle simulation dynamics p(s'|s, a) for each state-action pair in sim traj 
    ## to construct the transition dataset D_{s^hat}={(s^hat_i, a_i, s_{i+1})}
    O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state = \
        var_seq.preprocess_data(S_r=obs_real, O_r=imgs_real, A_r=acs_real, S_sim=obs_sim, A_sim=acs_sim, var_length=var_length,  var_length_sim=var_length_sim, all_hidden_state=all_hidden_state, all_cycle_hidden_state=all_cycle_hidden_state)

    ## train the discriminator
    discriminator_res_dict = {}
    discriminator_train_epochs = 10 if it == 0 else args.dis_train_epoch
    if (not too_strong_discriminator) or it % 20 == 0:
        for _ in range(discriminator_train_epochs):
            res = var_seq.train_discriminator(O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state)
            # accumulate the global dict
            for _key in res:
                discriminator_res_dict[_key] = discriminator_res_dict.get(_key, 0) + res[_key]
            # is discriminator too strong ?
            if args.gan_loss == GanLoss.MINIMAX:
                if res["dis_fake_prob"] < 0.4:
                    too_strong_discriminator = True
                else:
                    too_strong_discriminator = False
            else:
                too_strong_discriminator = False
                
        for _key in discriminator_res_dict:
            discriminator_res_dict[_key] /= discriminator_train_epochs
        logger.log_scalars("Discriminator", discriminator_res_dict, step=it)
        # logger.log_dict("Discriminator", discriminator_res_dict)
    else:
        logger.log_str(f"Skipping Discriminator training at Epoch {it}")

    ## train the generator
    mapping_res_dict = {}
    for _ in range(mapping_train_epoch):
        res = var_seq.train_mapping(O_r_rollout, O_r_first, A_r_rollout, A_r_first, A_sim_rollout, A_sim_first, S_sim_rollout, S_r_rollout, prev_hidden_state, prev_cycle_hidden_state)
        for _key in res:
            mapping_res_dict[_key] = mapping_res_dict.get(_key, 0) + res[_key]
    ### log mapping losses
    for _key in mapping_res_dict:
        mapping_res_dict[_key] /= mapping_train_epoch
    logger.log_scalars("Mapping", mapping_res_dict, step=it)
    # logger.log_dict("Mapping", mapping_res_dict)
            
    ## using the embeded dynamics model
    if args.emb_dynamic:
        obs_sim_dyn, _, acs_sim_dyn, _ = sample_sim_training_data(args, sim_training_dataset, OBS_BOUND, state_mean_std)
        ob_real2sim_trans = ob_real2sim.reshape((-1, ob_real2sim.shape[-1]))
        de_normalize_ob_trans = data_denormalization(ob_real2sim_trans, state_mean_std)
        if not is_robot_env:
            policy_infer_acs = model.step(de_normalize_ob_trans, deterministic=args.deter_policy)[0]

            policy_infer_acs = np.clip(policy_infer_acs, env.action_space.low, env.action_space.high) if args.clip_acs else policy_infer_acs
            policy_infer_acs = np.reshape(policy_infer_acs,  list(ob_real2sim.shape[:2]) + [acs_real.shape[-1]])

            acs_dyn_train = np.concatenate([acs_sim_dyn, acs_real, policy_infer_acs], axis=0)
            obs_dyn_train = np.concatenate([obs_sim_dyn, ob_real2sim.cpu().numpy(), ob_real2sim.cpu().numpy()], axis=0)
            obs_input, acs_input, next_obs_input, full_states, _ = safe_one_step_transition(args, env, is_robot_env, obs_dyn_train, acs_real, acs_dyn_train, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn, state_mean_std)
        ## update dynamics model
        if obs_input.shape[0] != 0:
            transition_train_res = transition_learner.update_transition(obs_input, acs_input, next_obs_input)
        logger.log_scalars("Embedded DM", transition_train_res, step=it)
        # logger.log_dict("Embedded DM", transition_train_res)
        transition_learner.copy_params_to_target()

    # evaluation
    eval_loss_dict = {
        "r2s_mse": compute_rmse(obs_real, ob_real2sim.cpu().numpy()), 
    }
    # eval_loss_dict["real2sim_loss"] = ((ob_real2sim.cpu().numpy() - obs_real)**2).sum() / var_length.sum().cpu().item()
    if it % 100 == 0:
        mapping_reward = []
        mapping_r2 = []
        mapping_r2_dynamics = []
        mapping_max_se_dynamics = []
        
        for _ in range(10):
            mapping_holder = MappingHolder(
                stack_imgs=args.stack_imgs, is_dynamics_emb = args.emb_dynamic, 
                var_seq = var_seq, default_rnn_state=var_seq.get_batch_zero_state(1), 
                obs_shape = env.state_space.shape[0], adjust_allowed=adjust_allowed,
                init_first_state = args.init_first_state, data_mean = state_mean_std[0], 
                data_std = state_mean_std[1] 
            )
            ret_dict = runner.run_traj(deter=False, mapping_holder=mapping_holder, render_img=True, run_in_realworld=True)
            while ret_dict[runner.TRAJ_LEN] == 0:
                ret_dict = runner.run_traj(deter=False, mapping_holder=mapping_holder, render_img=True, run_in_real_world=True)
            ob_traj = ret_dict[runner.OB_TRAJ]
            r2s_ob_traj = ret_dict[runner.R2S_OB_TRAJ]
            rew = ret_dict[runner.TOTAL_REW]
            ac_traj = ret_dict[runner.AC_TRAJ]
            
            mapping_r2.append(compute_adjusted_r2(ob_traj, r2s_ob_traj))
            mapping_reward.append(rew)
        
        eval_loss_dict = {
            "rew": np.mean(mapping_reward), 
            "rew_ratio": np.mean(mapping_reward) / expert_reward, 
            "mapping_r2": np.mean(mapping_r2)
        }
    
    logger.log_scalars("eval", eval_loss_dict, step=it)
            
            

