import numpy as np
from utils.util import *
from configs.config import *

def sample_sim_training_data(args, sim_training_dataset, OBS_BOUND, state_mean_std, traj_batch_size=None, raw_state=False):
    '''
    NOTE: all returns are numpy.ndarray
    '''
    traj_batch_size = args.trajectory_batch if traj_batch_size is None else traj_batch_size
    obs, _, acs, _, lengths = sim_training_dataset.get_next_batch(traj_batch_size)
    if args.clip_policy_bound:
        obs = np.clip(obs, -1.0 * OBS_BOUND, 1.0 * OBS_BOUND)
    if not raw_state:
        obs = data_normalization(obs, state_mean_std)
    # obs = obs.detach().cpu().numpy()
    return obs, None, acs, lengths

from UtilsRL.misc.decorator import profile
# @profile
def sample_next_batch_data(args, runner, expert_dataset, trajectory_buffer, OBS_BOUND, state_mean_std, type, iter, traj_batch_size=None):
    '''
    if loss type is VAE or GAN, we will directly use the expert_dataset, 
    else we will sample the dataset from the env
    NOTE: all returns are numpy.ndarray
    '''
    traj_batch_size = args.trajectory_batch if traj_batch_size is None else traj_batch_size
    MappingDirecition.config_check(args.mapping_direction)
    LossType.config_check(type)
    if (args.mapping_direction == MappingDirecition.RSR and type == LossType.VAE) or (args.mapping_direction == MappingDirecition.SRS and type == LossType.GAN):
        use_dataset = True
    else:
        use_dataset = False
    if not args.use_env_sample:
        use_dataset = True
        
    imgs = None
    if use_dataset:
            obs, imgs, acs, dones, lengths = expert_dataset.get_next_batch(traj_batch_size)
    else:
        if (iter % args.data_reused_times == 0 or not trajectory_buffer.can_sample(args.trajectory_batch * args.data_reused_times * 10)) and not trajectory_buffer.is_full():
            obs, acs, imgs, dones, rews, lengths = [], [], [], [], [], []
            for _ in range(traj_batch_size):
                ret_dict = runner.run_traj(deter=False, mapping_holder=None,
                                            render_img=False, run_in_realworld=False)
                while ret_dict[runner.TRAJ_LEN] == 0:
                    ret_dict = runner.run_traj(deter=False, mapping_holder=None,
                                                render_img=False, run_in_realworld=False)

                obs.append(ret_dict[runner.OB_TRAJ])
                acs.append(ret_dict[runner.AC_TRAJ])
                imgs.append(ret_dict[runner.IMG_TRAJ])
                dones.append(ret_dict[runner.DONE_TRAJ])
                lengths.append(ret_dict[runner.TRAJ_LEN])
                # rews = [ret_dict[runner.TOTAL_REW]]
                trajectory_buffer.add([ret_dict[runner.OB_TRAJ],
                                        ret_dict[runner.AC_TRAJ], ret_dict[runner.TRAJ_LEN]])
            obs = np.array(obs)
            acs = np.array(acs)
            imgs = np.array(imgs)
        else:
            obs, acs, lengths = trajectory_buffer.sample(traj_batch_size)
            # dones = None
    if args.clip_policy_bound:
        obs = np.clip(obs, -1.0 * OBS_BOUND, 1.0 * OBS_BOUND)
    obs = data_normalization(obs, state_mean_std)
    # obs = obs.detach().cpu().numpy()
    # dones = np.array(dones).astype(np.int)
    return obs, imgs, acs, lengths

@profile
def one_step_transition(args, env, is_robot_env, de_norm_obs_input, acs_real, acs_input):
    pass_one_step_transition_test = False
    de_normalize_next_obs, obs, full_states, acs = [], [], [], []

    for idx_selected in range(de_norm_obs_input.shape[0]):
        obs_trans = de_norm_obs_input[idx_selected].reshape([-1, de_norm_obs_input.shape[-1]])
        acs_trans = acs_input[idx_selected].reshape([-1, acs_real.shape[-1]])
        for idx in range(de_norm_obs_input.shape[1] - 1):  # skip the zero time-step.
            de_normalize_ob = obs_trans[idx]
            ac = acs_trans[idx]
            if np.all(de_normalize_ob == 0):
                idx -= 1
                break
            try:
                de_normalize_next_ob, r, d, info = env.set_ob_and_step(de_normalize_ob, ac, ret_full_state=is_robot_env)
                full_state = info['full_state']
            except Exception as e:
                break
                # raise RuntimeError
            if np.random.random() < 0.1 and not pass_one_step_transition_test:
                pass_one_step_transition_test = True
            obs.append(de_normalize_ob)
            acs.append(ac)
            full_states.append(full_state)
            de_normalize_next_obs.append(de_normalize_next_ob)
        if len(obs) > args.dyn_batch_size:
            break
    de_normalize_next_obs = np.array(de_normalize_next_obs)
    next_obs = de_normalize_next_obs
    obs = np.array(obs)
    acs = np.array(acs)
    full_states = np.array(full_states)
    assert np.where(np.isnan(obs))[0].shape[0] == 0
    assert np.where(np.isnan(next_obs))[0].shape[0] == 0
    return obs, acs, next_obs, full_states, idx

def safe_one_step_transition(args, env, is_robot_env, norm_obs_input, acs_real, raw_acs_input, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn, state_mean_std):
    norm_obs_input_clip = np.clip(norm_obs_input, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn)
    obs_input, acs_input, next_obs_input, full_states, end_idx = one_step_transition(args, env, is_robot_env, data_denormalization(norm_obs_input_clip, state_mean_std), acs_real, raw_acs_input)
    obs_input = data_normalization(obs_input, state_mean_std)
    next_obs_input = data_normalization(next_obs_input, state_mean_std)
    next_obs_input = np.clip(next_obs_input, update_dynamics_range_min_trans_learn, update_dynamics_range_max_trans_learn)
    return obs_input, acs_input, next_obs_input, full_states, end_idx

def obs_acs_reshape(obs_input, acs_input):
    # obs_input = obs_input.detach().cpu().numpy()
    # acs_input = acs_input.detach().cpu().numpy()
    # alignment
    obs_train = obs_input[:, :-1].reshape([-1, obs_input.shape[-1]])
    acs_train = acs_input[:, :-1].reshape([-1, acs_input.shape[-1]])
    obs_next_train = obs_input[:, 1:].reshape([-1, obs_input.shape[-1]])
    # remove mask states.
    not_done_idx = np.where(np.any(obs_train != 0, axis=1))
    obs_train = obs_train[not_done_idx]
    acs_train = acs_train[not_done_idx]
    obs_next_train = obs_next_train[not_done_idx]
    not_done_idx = np.where(np.any(obs_next_train != 0, axis=1))
    obs_train = obs_train[not_done_idx]
    acs_train = acs_train[not_done_idx]
    obs_next_train = obs_next_train[not_done_idx]
    return obs_train, acs_train, obs_next_train