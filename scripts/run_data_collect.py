import argparse
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
import pickle
import os.path as osp
import tqdm
from typing import Union
import gym

from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback, BaseCallback
from stable_baselines.common.vec_env import sync_envs_normalization, VecEnv

from RLA.easy_log.tester import tester
from RLA.easy_log import logger

# from codas.utils.functions import *
# from codas.utils.config import *
# from codas.wrapper.env_wrapper import GeneratorWrapper, is_dapg_env, make_vec_env
# from codas.wrapper.policy_wrapper import WrappedPolicy
# from codas.train.rollout import Runner
# from codas.rl.ppo2 import PPO2
from utils.functions import *
from utils.ppo2 import PPO2
from utils.env_wrapper import GeneratorWrapper, is_dapg_env, make_vec_env
from utils.rollout import Runner
from utils.policy_wrapper import PolicyWrapper


from configs.env_config_map import env_config_map
from configs.config import *

import reset_able_mj_env
import ast

def get_param():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Variational Sequence")
    parser.add_argument('--seed', help='RNG seed', type=int, default=88)
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--expert_root', type=str, default=DATA_ROOT)
    parser.add_argument('--task', type=str, default='data_collect')
    # new params
    parser.add_argument('--env_id', help='environment ID', default='InvertedDouble-v5')
    parser.add_argument('--num_env', default=1, type=int)

    parser.add_argument('--policy_timestep', type=int, default=1000000)
    parser.add_argument('--collect_trajs', type=int, default=COLLECT_TRAJ)
    parser.add_argument('--max_sequence', type=int, default=1002)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--dynamic_param', type=float, default=1.0)

    parser.add_argument('--deter', type=ast.literal_eval, default=False)
    parser.add_argument('--auto_env_map', type=ast.literal_eval, default=True)
    parser.add_argument('--log_tb', type=ast.literal_eval, default=False)
    parser.add_argument('--log_dir', type=str, default="../log/")
    parser.add_argument('--start_fraction', type=float, default=0.05)
    parser.add_argument('--end_fraction', type=float, default=1.0)
    parser.add_argument('--trajs_per_callback', type=int, default=2)
    parser.add_argument('--tot_training_trajs', type=int, default=COLLECT_SIM_TRAJ)
    parser.add_argument('--stoc_init_range', type=float, default=0.005)
    args = parser.parse_args()
    kwargs = vars(args)
    if kwargs['auto_env_map'] and kwargs['env_id'] in env_config_map:
        kwargs.update(env_config_map[kwargs['env_id']])
    args = argparse.Namespace(**kwargs)
    return args


class CollectStateCallback(BaseCallback):
    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 trajs_per_callback: int = 5,
                 tot_steps: int = 1000000, start_fraction: float = 0.5,
                 end_fraction: float = 0.5, tot_trajs_to_collect=600,
                 deterministic: bool = True,
                 max_horizon: int = 1002,
                 output_path: str = "/tmp/codas_callback_cache.npz", verbose: int = 1):
        super(CollectStateCallback, self).__init__(verbose=verbose)
        self.tot_steps = tot_steps
        self.start_fraction = start_fraction
        self.output_path = output_path
        self.traj_obs = []
        self.traj_action = []
        self.traj_length = []
        self.traj_return = []
        self.collected_traj_lengths = []
        self.eval_env = eval_env
        self.deterministic = deterministic
        self.tot_trajs_to_collect = tot_trajs_to_collect
        self.max_horizon = max_horizon
        # calculate collect interval and trajs per collection
        num_validate_callbacks = tot_trajs_to_collect // trajs_per_callback
        validate_fraction = max(0, end_fraction - start_fraction)
        self.callback_interval = int((tot_steps * validate_fraction) // num_validate_callbacks)
        self.start_callback_step = int(tot_steps * start_fraction)
        self.start_callback_id = 0
        self.final_callback_id = num_validate_callbacks

        self.trajs_per_callback = trajs_per_callback
        self.curr_callback_id = 0
        print("\033[32msetting data collection callback\033[0m:\n"
              "\ttotal callbacks:{}\n"
              "\tstarting timestep:{}\n"
              "\teval freq:{}\n"
              "\ttrajs_per_callback:{}".format(
            self.final_callback_id, self.start_callback_step, self.callback_interval, trajs_per_callback))

    def _on_step(self):
        curr_timestep = self.n_calls + 1
        # if curr_timestep > 8192:
        # tester.time_step_holder.set_time(curr_timestep)
        if curr_timestep <= self.start_callback_step or \
                ((curr_timestep - self.start_callback_step) % self.callback_interval) != 0 \
                or self.curr_callback_id >= self.final_callback_id:
            return True
        # collect trajectories
        self.curr_callback_id += 1
        if self.curr_callback_id < self.final_callback_id:
            num_trajs_to_collect = self.trajs_per_callback
        else:
            num_trajs_to_collect = self.tot_trajs_to_collect - len(self.traj_obs)
        obs, actions, lengths, returns = self.collect_trajs(num_trajs_to_collect)
        self.traj_obs += obs
        self.traj_action += actions
        self.traj_length += lengths
        self.traj_return += returns
        if self.curr_callback_id >= self.final_callback_id:
            # write collected trajs to temp file
            if len(self.traj_obs) > self.tot_trajs_to_collect:
                self.traj_obs = self.traj_obs[:self.tot_trajs_to_collect]
                self.traj_length = self.traj_length[:self.tot_tra]
            np.savez(self.output_path, obs=self.traj_obs, acs=self.traj_action, traj_len=self.traj_length,
                     ep_rets=self.traj_return)
            print("saving trajs:", len(self.traj_obs), self.traj_obs[0].shape, "to", self.output_path)
        return True

    def collect_trajs(self, num_trajs):
        if num_trajs <= 0:
            return []
        tot_obs = []
        ep_rewards = []
        ep_lengths = []
        tot_actions = []
        sync_envs_normalization(self.training_env, self.eval_env)
        epsilon = self.eval_env.epsilon
        obs_mean = self.eval_env.obs_rms.mean
        obs_var = self.eval_env.obs_rms.var
        obs_std = np.sqrt(obs_var + epsilon)
        ret_mean = self.eval_env.ret_rms.mean
        ret_var = self.eval_env.ret_rms.var
        # obs_std = np.sqrt(obs_var + epsilon)
        ret_std = np.sqrt(ret_var + epsilon)
        for i in range(num_trajs):
            # if not isinstance(self.eval_env, VecEnv) or i == 0:
            obs = self.eval_env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            obs_traj = np.zeros((self.max_horizon, self.eval_env.observation_space.shape[0]), dtype=np.float64)
            ac_traj = np.zeros((self.max_horizon, self.eval_env.action_space.shape[0]), dtype=np.float64)
            for j in range(self.max_horizon):
                action, state = self.model.predict(obs, state=state, deterministic=self.deterministic)
                # denormalize obs [unnecessary now]
                raw_obs = obs  # (obs * obs_std) + obs_mean
                obs_traj[j] = raw_obs
                ac_traj[j] = action
                obs, reward, done, _info = self.eval_env.step(action)
                raw_reward = reward * ret_std
                episode_reward += raw_reward
                episode_length += 1
                if done:
                    break
            tot_obs.append(obs_traj)
            tot_actions.append(ac_traj)
            ep_rewards.append(episode_reward)
            ep_lengths.append(episode_length)
        logger.record_tabular("perf/ret", np.mean(ep_rewards))
        logger.dump_tabular()
        if self.verbose > 0:
            print("Callback {}/{}:\t average return: {:.01f}\t average length:{:.01f}".format(self.curr_callback_id,
                                                                                              self.final_callback_id,
                                                                                              np.mean(ep_rewards),
                                                                                              np.mean(ep_lengths)))
        return tot_obs, tot_actions, ep_lengths, ep_rewards


def main():
    args = get_param()
    kwargs = vars(args)
    # tester.set_hyper_param(**kwargs)
    # tester.add_record_param(['info',
                            #  "seed",
                            #  "env_id", "policy_timestep"])

    def get_package_path():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # tester.configure(task_name='data_collect',
                    #  private_config_path=os.path.join(get_package_path(), 'rla_config.yaml'),
                    #  log_root=get_package_path())
    # tester.log_files_gen()
    # tester.print_args()

    load_path = osp.join(DATA_ROOT, "saved_model")
    set_global_seeds(args.seed)
    model_path = osp.join(load_path, "ppo_{}_{}_full.zip".format(args.env_id, args.policy_timestep))
    env_path = osp.join(load_path, "{}_full".format(args.env_id))
    if not (osp.exists(model_path) and osp.exists(env_path)):
        print("The pre-trained target-domain policy cannot be found.  We will train a new target-domain policy. ")
        sched_lr = LinearSchedule(args.policy_timestep, 0., 3e-4)
        # stoc_init_range is set to the default value of the real-world environment.
        env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=0.005)
        env = VecNormalize(env, norm_obs=False)
        if args.log_tb:
            tb_log_dir = args.log_dir
        else:
            tb_log_dir = None
        model = PPO2(policy=MlpPolicy, env=env, verbose=1, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99,
                     noptepochs=10, ent_coef=0.0, learning_rate=sched_lr.value, cliprange=0.2,
                     tensorboard_log=tb_log_dir)
        model.learn(total_timesteps=args.policy_timestep)
        model.save(model_path)
        env.save(env_path)

    print("full data collection")
    # train a ppo policy from scratch in the source domain
    sched_lr = LinearSchedule(args.policy_timestep, 0., 3e-4)
    if args.log_tb:
        tb_log_dir = args.log_dir
    else:
        tb_log_dir = None
    n_steps = 2048
    args.policy_timestep = int((args.policy_timestep // n_steps) * n_steps)
    sim_env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=args.stoc_init_range)
    sim_env = VecNormalize(sim_env, norm_obs=False)
    eval_env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=args.stoc_init_range)
    eval_env = VecNormalize(eval_env, norm_obs=False, training=False)
    model = PPO2(policy=MlpPolicy, env=sim_env, verbose=1, n_steps=n_steps, nminibatches=32, lam=0.95, gamma=0.99,
                 noptepochs=10, ent_coef=0.0, learning_rate=sched_lr.value, cliprange=0.2, tensorboard_log=tb_log_dir)
    tmp_file_path = "/tmp/codas_callback_{}_cache.npz".format(args.env_id)
    callback = CollectStateCallback(eval_env, trajs_per_callback=args.trajs_per_callback,
                                    tot_steps=args.policy_timestep,
                                    start_fraction=args.start_fraction,
                                    end_fraction=args.end_fraction,
                                    tot_trajs_to_collect=args.tot_training_trajs,
                                    deterministic=False, max_horizon=args.max_sequence,
                                    output_path=tmp_file_path, verbose=1)
    model.learn(total_timesteps=args.policy_timestep, callback=callback)
    # read obs from tmp file and delete tmp file
    data = np.load(tmp_file_path)
    train_obs = data['obs']
    train_acs = data['acs']
    train_traj_len = data['traj_len']
    train_ep_rets = data['ep_rets']
    os.system("rm {}".format(tmp_file_path))

    # collect the target-domain dataset.
    print("loaded pre-trained policy from {}".format(load_path))
    model = PPO2.load(model_path)
    real_env = make_vec_env(args.env_id, num_env=args.num_env, dynamic_param=args.dynamic_param, stoc_init_range=0.005)
    real_env = VecNormalize.load(env_path, real_env)
    real_env.training = False
    real_env.norm_reward = False
    real_env = GeneratorWrapper(real_env)

    img_shape = {ImgShape.WIDTH: args.image_size, ImgShape.HEIGHT: args.image_size, ImgShape.CHANNEL: 3}
    runner = Runner(simulator_env=None, real_world_env=real_env, max_horizon=args.max_sequence, img_shape=img_shape,
                    clip_acs=False, real_policy=model, sim_policy=None, exact_consist=False)

    obs_acs = {"obs": [], "acs": [], "ep_rets": [], "imgs": [], 'ac_means': [], 'traj_len': []}

    model_name = str(model_path).split('/')[-1].split('.')[0]
    tot_rews = []
    for _ in tqdm.tqdm(range(args.collect_trajs)):
        ret_dict = runner.run_traj(deter=False, mapping_holder=None, render_img=True, run_in_realworld=True)
        total_rew = ret_dict[runner.TOTAL_REW]
        ob_traj = ret_dict[runner.OB_TRAJ]
        ac_traj = ret_dict[runner.AC_TRAJ]
        img_traj = ret_dict[runner.IMG_TRAJ]
        traj_len = ret_dict[runner.TRAJ_LEN]
        tot_rews.append(total_rew)
        print(total_rew, traj_len)
        img_traj = (img_traj * 255.0).astype(np.uint8)
        obs_acs['obs'].append(ob_traj)
        obs_acs['acs'].append(ac_traj)
        obs_acs['ep_rets'].append(total_rew)
        obs_acs['imgs'].append(img_traj)
        obs_acs['traj_len'].append(traj_len)

    output_path = osp.join(args.expert_root, model_name + '_' +
                           str(args.collect_trajs) + '_deter_' + str(args.deter) + '_uint8_full.npz')

    print(args.env_id, np.mean(np.array(tot_rews)), np.std(np.array(tot_rews)))


    np.savez(output_path,
             obs=obs_acs['obs'], acs=obs_acs['acs'], ep_rets=obs_acs['ep_rets'], imgs=obs_acs['imgs'],
             traj_len=obs_acs['traj_len'], train_obs=train_obs, train_acs=train_acs,
             train_ep_rets=train_ep_rets, train_traj_len=train_traj_len)
    print("---------- done -------------")


if __name__ == '__main__':
    main()
