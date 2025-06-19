# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import dataclasses
import os
import pickle
import time
import json
from tqdm import tqdm
os.environ["MUJOCO_GL"] = "egl"  # 推荐优先尝试 egl

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import env_preprocessing
import MRQ
import utils


@dataclasses.dataclass
class DefaultExperimentArguments:
    Atari_total_timesteps: int = 25e5
    Atari_eval_freq: int = 1e5

    Dmc_total_timesteps: int = 5e5
    Dmc_eval_freq: int = 5e3

    Gym_total_timesteps: int = 1e6
    Gym_eval_freq: int = 5e3

    def __post_init__(self): utils.enforce_dataclass_type(self)


def main():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--env', default='Gym-HalfCheetah-v4', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--total_timesteps', default=-1, type=int) # Uses default, input to override.
    parser.add_argument('--device', default='cuda', type=str)
    # Evaluation
    parser.add_argument('--eval_freq', default=-1, type=int) # Uses default, input to override.
    parser.add_argument('--eval_eps', default=10, type=int)
    # File name and locations
    parser.add_argument('--project_name', default='', type=str) # Uses default, input to override.
    parser.add_argument('--eval_folder', default='./evals', type=str)
    parser.add_argument('--log_folder', default='./logs', type=str)
    parser.add_argument('--save_folder', default='./checkpoint', type=str)
    # Experiment checkpointing
    parser.add_argument('--save_experiment', default=False, action=argparse.BooleanOptionalAction, type=bool)
    parser.add_argument('--save_freq', default=1e5, type=int)
    parser.add_argument('--load_experiment', default=False, action=argparse.BooleanOptionalAction, type=bool)
    # 新增：优先经验回放开关
    parser.add_argument('--use_prio_buffer', action='store_true', help='Use prioritized replay buffer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    default_arguments = DefaultExperimentArguments()
    env_type = args.env.split('-',1)[0]
    if args.total_timesteps == -1: args.total_timesteps = default_arguments.__dict__[f'{env_type}_total_timesteps']
    if args.eval_freq == -1: args.eval_freq = default_arguments.__dict__[f'{env_type}_eval_freq']

    # File name and make folders
    if args.project_name == '':
        args.project_name = f'MRQ+{args.env}+{args.seed}'
        if args.use_prio_buffer:
            args.project_name += '+PRIO'
    if not os.path.exists(args.eval_folder): os.makedirs(args.eval_folder)
    if not os.path.exists(args.log_folder): os.makedirs(args.log_folder)
    if args.save_experiment and not os.path.exists(f'{args.save_folder}/{args.project_name}'):
        os.makedirs(f'{args.save_folder}/{args.project_name}')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.load_experiment:
        exp = load_experiment(args.save_folder, args.project_name, device, args)
    else:
        env = env_preprocessing.Env(args.env, args.seed, eval_env=False)
        eval_env = env_preprocessing.Env(args.env, args.seed+100, eval_env=True) # +100 to make sure the seed is different.

        agent = MRQ.Agent(env.obs_shape, env.action_dim, env.max_action,
            env.pixel_obs, env.discrete, device, env.history, use_prio_buffer=args.use_prio_buffer)

        # 创建TensorBoard日志记录器
        writer = SummaryWriter(log_dir=os.path.join(args.log_folder, args.project_name))
        
        # 记录实验配置
        config = {
            'algorithm': agent.name,
            'env': env.env_name,
            'seed': env.seed,
            'obs_shape': env.obs_shape,
            'action_dim': env.action_dim,
            'discrete_actions': env.discrete,
            'pixel_obs': env.pixel_obs,
            'total_timesteps': args.total_timesteps,
            'eval_freq': args.eval_freq,
            'eval_eps': args.eval_eps,
            'use_prio_buffer': args.use_prio_buffer,
            **dataclasses.asdict(agent.hp)
        }
        
        # 保存配置到JSON文件
        config_path = os.path.join(args.log_folder, args.project_name, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
        # 记录配置到TensorBoard（使用text而不是hparams）
        config_text = '\n'.join([f'{k}: {v}' for k, v in config.items()])
        writer.add_text('config', config_text, 0)

        exp = OnlineExperiment(agent, env, eval_env, writer, [],
            0, args.total_timesteps, 0,
            args.eval_freq, args.eval_eps, args.eval_folder, args.project_name,
            args.save_experiment, args.save_freq, args.save_folder)

    exp.run()


class OnlineExperiment:
    def __init__(self, agent: object, env: object, eval_env: object, writer: SummaryWriter, evals: list,
            t: int, total_timesteps: int, time_passed: float,
            eval_freq: int, eval_eps: int, eval_folder: str, project_name: str,
            save_full: bool=False, save_freq: int=1e5, save_folder: str=''):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.evals = evals

        self.writer = writer

        self.t = t
        self.time_passed = time_passed
        self.start_time = time.time()

        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.eval_eps = eval_eps

        self.eval_folder = eval_folder
        self.project_name = project_name
        self.save_full = save_full
        self.save_freq = save_freq
        self.save_folder = save_folder

        self.init_timestep = True


    def run(self):
        state = self.env.reset()
        pbar = tqdm(total=self.total_timesteps, initial=self.t, desc='Training')
        while self.t <= self.total_timesteps:
            self.maybe_evaluate()
            if self.save_full and self.t % self.save_freq == 0 and not self.init_timestep:
                save_experiment(self)

            action = self.agent.select_action(np.array(state))
            if action is None: action = self.env.action_space.sample()

            next_state, reward, terminated, truncated = self.env.step(action)
            self.agent.replay_buffer.add(state, action, next_state, reward, terminated, truncated)
            state = next_state

            self.agent.train()

            if terminated or truncated:
                # 记录每个episode的信息
                self.writer.add_scalar('train/episode_reward', self.env.ep_total_reward, self.env.ep_num)
                self.writer.add_scalar('train/episode_length', self.env.ep_timesteps, self.env.ep_num)
                self.writer.add_scalar('train/total_timesteps', self.t + 1, self.env.ep_num)
                
                pbar.set_postfix({
                    'episode': self.env.ep_num,
                    'ep_len': self.env.ep_timesteps,
                    'reward': f'{self.env.ep_total_reward:.3f}'
                })
                state = self.env.reset()

            self.t += 1
            pbar.update(1)
            self.init_timestep = False
        
        pbar.close()


    def maybe_evaluate(self):
        if self.t % self.eval_freq != 0:
            return

        # We save after evaluating, this avoids re-evaluating immediately after loading an experiment.
        if self.t != 0 and self.init_timestep:
            return

        total_reward = np.zeros(self.eval_eps)
        for ep in tqdm(range(self.eval_eps), desc='Evaluating', leave=False):
            state, terminated, truncated = self.eval_env.reset(), False, False
            while not (terminated or truncated):
                action = self.agent.select_action(np.array(state), use_exploration=False)
                state, _, terminated, truncated = self.eval_env.step(action)
            total_reward[ep] = self.eval_env.ep_total_reward

        mean_reward = total_reward.mean()
        self.evals.append(mean_reward)

        # 记录评估结果
        self.writer.add_scalar('eval/mean_reward', mean_reward, self.t)
        self.writer.add_scalar('eval/std_reward', total_reward.std(), self.t)
        self.writer.add_scalar('time/total_minutes', 
                             (time.time() - self.start_time + self.time_passed)/60., 
                             self.t)

        print(f'\nEvaluation at {self.t} time steps\n'
              f'Average total reward over {self.eval_eps} episodes: {mean_reward:.3f}\n'
              f'Total time passed: {round((time.time() - self.start_time + self.time_passed)/60., 2)} minutes')

        np.savetxt(f'{self.eval_folder}/{self.project_name}.txt', self.evals, fmt='%.14f')


def save_experiment(exp: OnlineExperiment):
    # Save experiment settings
    exp.time_passed += time.time() - exp.start_time
    var_dict = {k: exp.__dict__[k] for k in ['t', 'eval_freq', 'eval_eps']}
    var_dict['time_passed'] = exp.time_passed + time.time() - exp.start_time
    var_dict['np_seed'] = np.random.get_state()
    var_dict['torch_seed'] = torch.get_rng_state()
    np.save(f'{exp.save_folder}/{exp.project_name}/exp_var.npy', var_dict)
    # Save eval
    np.savetxt(f'{exp.save_folder}/{exp.project_name}.txt', exp.evals, fmt='%.14f')
    # Save envs
    pickle.dump(exp.env, file=open(f'{exp.save_folder}/{exp.project_name}/env.pickle', 'wb'))
    pickle.dump(exp.eval_env, file=open(f'{exp.save_folder}/{exp.project_name}/eval_env.pickle', 'wb'))
    # Save agent
    exp.agent.save(f'{exp.save_folder}/{exp.project_name}')

    print('Saved experiment')


def load_experiment(save_folder: str, project_name: str, device: torch.device, args: object):
    # Load experiment settings
    exp_dict = np.load(f'{save_folder}/{project_name}/exp_var.npy', allow_pickle=True).item()
    # This is not sufficient to guarantee the experiment will run exactly the same,
    # however, it does mean the original seed is not reused.
    np.random.set_state(exp_dict['np_seed'])
    torch.set_rng_state(exp_dict['torch_seed'])
    # Load eval
    evals = np.loadtxt(f'{save_folder}/{project_name}.txt').tolist()
    # Load envs
    env = pickle.load(open(f'{save_folder}/{project_name}/env.pickle', 'rb'))
    eval_env = pickle.load(open(f'{save_folder}/{project_name}/eval_env.pickle', 'rb'))
    # Load agent
    agent_dict = np.load(f'{save_folder}/{project_name}/agent_var.npy', allow_pickle=True).item()
    agent = MRQ.Agent(env.obs_shape, env.action_dim, env.max_action,
        env.pixel_obs, env.discrete, device, env.history, dataclasses.asdict(agent_dict['hp']))
    agent.load(f'{save_folder}/{project_name}')

    # 创建TensorBoard日志记录器
    writer = SummaryWriter(log_dir=os.path.join(args.log_folder, args.project_name))
    
    # 加载并记录配置
    config_path = os.path.join(args.log_folder, args.project_name, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        config_text = '\n'.join([f'{k}: {v}' for k, v in config.items()])
        writer.add_text('config', config_text, 0)
    
    print(f'Loaded experiment\nStarting from: {exp_dict["t"]} time steps.')

    return OnlineExperiment(agent, env, eval_env, writer, evals,
        exp_dict['t'], args.total_timesteps, exp_dict['time_passed'],
        exp_dict['eval_freq'], exp_dict['eval_eps'], args.eval_folder, args.project_name,
        args.save_experiment, args.save_freq, args.save_folder)


if __name__ == '__main__':
    main()
