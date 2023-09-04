import argparse
import sys
sys.path.append("../")
from .env_config_map import env_config_map
from .config import *
import os.path as osp

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    import ast
    # task info
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--env_id', help='environment ID', default='InvertedDouble-v5')
    parser.add_argument('--auto_env_map', type=ast.literal_eval, default=True)
    parser.add_argument('--pretrain_path', type=str, default=osp.join(DATA_ROOT, 'saved_model/transition_weights.npy'))
    parser.add_argument('--pretrain_mean_std', type=str, default=osp.join(DATA_ROOT, 'saved_model/state_mean_std.npy'))
    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=COLLECT_TRAJ)
    parser.add_argument('--alg_type', type=str, default=AlgType.CODAS)
    parser.add_argument('--cycle_loss', type=ast.literal_eval, default=False)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--expert_perf_eval_times', type=int, default=100)
    parser.add_argument('--action_noise_level', type=float, default=0.0)
    parser.add_argument('--dynamic_param', type=float, default=1.0)
    parser.add_argument('--load_data', type=str, default='')
    parser.add_argument('--load_task', type=str, default='')
    parser.add_argument('--max_sequence', type=int, default=500)
    parser.add_argument("--max_tf_util", help="per process gpu memory fraction fot tf", type=float, default=1.0)
    # reduce the rollout step of traj-discriminator will make the training process more stable
    # if the size of dataset is not enough.
    parser.add_argument('--rollout_step', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=-1)
    parser.add_argument('--dis_test', type=ast.literal_eval, default=False)
    parser.add_argument('--deter_policy', type=ast.literal_eval, default=False)
    parser.add_argument('--label_image_test', type=ast.literal_eval, default=False)
    parser.add_argument('--dynamic_test', type=ast.literal_eval, default=False)
    parser.add_argument('--use_dataset_mean_std', type=ast.literal_eval, default=False)
    parser.add_argument('--exact_consist', type=ast.literal_eval, default=False)
    # transformation setting
    parser.add_argument('--ob_transformation', type=str, default=Transformation.IMG)
    # params added for image_input
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--output_image', type=ast.literal_eval, default=True)
    # learn expert policy
    # network params
    parser.add_argument('--act_fn', type=str, default='LeakyReLU')
    parser.add_argument('--dyn_act_fn', type=str, default='Tanh')
    parser.add_argument('--layer_norm', type=ast.literal_eval, default=True)
    parser.add_argument('--safe_log', type=ast.literal_eval, default=False)
    parser.add_argument('--mapping_direction', type=str, default=MappingDirecition.RSR)
    parser.add_argument('--stack_imgs', type=int, default=1)
    # 1. discriminator params
    parser.add_argument('--dis_struc', type=str, default=DiscriminatorStructure.OB_AC_CONCATE)
    parser.add_argument('--rnn_cell', type=str, default='GRU')
    parser.add_argument('--disc_hid_dims', type=ast.literal_eval, default=[256, 256, 256])
    parser.add_argument('--disc_img_hid_dims', type=ast.literal_eval, default=[256])
    parser.add_argument('--dyn_hid_dims', type=ast.literal_eval, default=[512, 512, 512])
    parser.add_argument('--disc_emb_hid_dim', type=int, default=256)
    parser.add_argument('--num_env', type=int, default=1)
    # 2. embedding params
    parser.add_argument('--emb_hid_dims', type=ast.literal_eval, default=[256, 256, 256, 256])
    parser.add_argument('--emb_output_size', type=int, default=256)
    parser.add_argument('--mlp', type=ast.literal_eval, default=False)
    parser.add_argument('--clip_acs', type=ast.literal_eval, default=True)
    # 3. encode param.
    parser.add_argument('--gan_loss', type=str, default=GanLoss.MINIMAX)
    parser.add_argument('--r2s_rnn_hid_dims', type=ast.literal_eval, default=[128, 128])
    parser.add_argument('--r2s_output_hid_dims', nargs='+', type=int, default=[])
    parser.add_argument('--adjust_allowed', type=float, default=1.0)
    parser.add_argument('--emb_dynamic', type=ast.literal_eval, default=True)
    parser.add_argument('--policy_infer_transition', type=ast.literal_eval, default=True)
    # 3. reconstruction params
    parser.add_argument('--s2r_hid_dims', type=ast.literal_eval, default=[256, 256, 256, 256])
    parser.add_argument('--s2r_rnn_hid_dims', type=ast.literal_eval, default=[])
    parser.add_argument('--s2r_emb_dim', type=int, default=256)
    parser.add_argument('--reconstruct_clip', type=float, default=-1)
    parser.add_argument('--res_struc', type=str, default=ResnetStructure.EMBEDDING_RAS)
    parser.add_argument('--resc_act_fn', type=str, default='Identity')
    parser.add_argument('--real_ob_input', type=ast.literal_eval, default=False)
    parser.add_argument('--hard_training', type=ast.literal_eval, default=False)
    parser.add_argument('--retrain_dynamics', type=ast.literal_eval, default=False)
    parser.add_argument('--filter_traj', type=ast.literal_eval, default=False)
    # learning params
    parser.add_argument('--lr_gen', type=float, default=0.0001)
    parser.add_argument('--lr_dyn', type=float, default=0.0001)
    parser.add_argument('--dyn_lr_pretrain', type=float, default=0.0001)
    parser.add_argument('--dyn_l2_loss', type=float, default=0.0000002)
    parser.add_argument('--mapping_l2_loss', type=float, default=0.0)
    parser.add_argument('--dis_l2_loss', type=float, default=0.0)
    parser.add_argument('--lr_dis', type=float, default=0.00005)
    parser.add_argument('--lr_rescale', type=float, default=1.0)
    parser.add_argument('--dyn_batch_size', type=int, default=1024)
    parser.add_argument('--mapping_train_epoch', type=int, default=5)
    parser.add_argument('--dis_train_epoch', type=int, default=1)
    parser.add_argument('--trajectory_batch', type=int, default=10)
    parser.add_argument('--decay_ratio', type=float, default=0.5)
    parser.add_argument('--total_timesteps', type=int, default=40000)
    parser.add_argument('--lambda_a', type=float, default=2)
    parser.add_argument('--lambda_b', type=float, default=0.1)
    parser.add_argument('--norm_std_bound', type=float, default=0.05)
    parser.add_argument('--stoc_init_range', type=float, default=0.005)
    parser.add_argument('--grad_clip_norm', type=float, default=10)
    parser.add_argument('--random_set_to_zero', type=ast.literal_eval, default=False)
    parser.add_argument('--data_normalize', type=ast.literal_eval, default=True)
    parser.add_argument('--minmax_normalize', type=ast.literal_eval, default=False)
    parser.add_argument('--npmap_replace', type=ast.literal_eval, default=False) # namap will reduce the occupancy of mermory.
    parser.add_argument('--merge_d_train', type=ast.literal_eval, default=True)
    parser.add_argument('--traj_dis', type=ast.literal_eval, default=False)
    parser.add_argument('--clip_policy_bound', type=ast.literal_eval, default=True)
    # TODO： the correctness of related code should be check.
    parser.add_argument('--init_first_state', type=ast.literal_eval, default=False)
    # trajectory-buffer hyperparameter
    parser.add_argument('--use_env_sample', type=ast.literal_eval, default=True)
    parser.add_argument('--do_save_checkpoint', type=ast.literal_eval, default=True)
    parser.add_argument('--pool_size', type=int, default=6000)
    parser.add_argument('--data_reused_times', type=int, default=10)
    # ablation study
    parser.add_argument('--data_used_fraction', type=float, default=1)
    parser.add_argument("--use_noise_env", type=ast.literal_eval, default=False)
    parser.add_argument("--dual_policy_noise_std", help="use obs collected by noise action", type=float, default=0.0)
    # policy trainable
    parser.add_argument('--policy_trainable', type=bool, default=False, help="whether policy is trainable during training")

    ######## tensorboard logger ########
    parser.add_argument('--exp_name', type=str, default="debug", help="experiment name, will be used in tensorboard logger")
    parser.add_argument('--log_dir', type=str, default="tb", help="the dir which tensorboard logger will log to")

    args = parser.parse_args()
    kwargs = vars(args)
        # kwargs['exact_consist'] = True
    if kwargs['auto_env_map'] and kwargs['env_id'] in env_config_map:
        kwargs.update(env_config_map[kwargs['env_id']])
    assert kwargs['max_sequence'] % kwargs['rollout_step'] == 0
    # seq_length = int(np.ceil(args.max_sequence / args.rollout_step) * args.rollout_step)
    if kwargs['alg_type'] == AlgType.VAN_GAN:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = True
        kwargs['cycle_loss'] = False
    elif kwargs['alg_type'] == AlgType.CYCLE_GAN:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = True
        kwargs['cycle_loss'] = True
    elif kwargs["alg_type"] == AlgType.VAN_GAN_STACK:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = True
        kwargs["stack_imgs"] = 4
    elif kwargs['alg_type'] == AlgType.VAN_GAN_RNN:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['lambda_b'] = 0
        kwargs['mlp'] = False
    elif kwargs['alg_type'] == AlgType.CODAS:
        kwargs['traj_dis'] = True
    elif kwargs['alg_type'] == AlgType.NO_DYN:
        kwargs['traj_dis'] = True
        kwargs['emb_dynamic'] = False
    elif kwargs['alg_type'] == AlgType.NO_DYN_NO_TRAJ_DIS:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
    elif kwargs['alg_type'] == AlgType.MLP:
        kwargs['traj_dis'] = False
        kwargs['emb_dynamic'] = False
        kwargs['mlp'] = True
    elif kwargs['alg_type'] == AlgType.MLP_TRAJ_DIS:
        kwargs['traj_dis'] = True
        kwargs['emb_dynamic'] = False
        kwargs['mlp'] = True
        # kwargs["r2s_output_hid_dims"] = kwargs["dyn_hid_dims"]
    elif kwargs['alg_type'] == AlgType.NO_TRAJ_DIS:
        kwargs['traj_dis'] = False
    else:
        raise NotImplementedError

    kwargs["lr_dis"] *= kwargs["lr_rescale"]
    kwargs["lr_gen"] *= kwargs["lr_rescale"]
    args = argparse.Namespace(**kwargs)
    return args

if __name__ == "__main__":
    args = get_param()
    print(type(args))
    print(vars(args))