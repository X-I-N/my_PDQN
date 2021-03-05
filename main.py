import os
import numpy as np
import gym
import torch
import datetime
import argparse
import gym_soccer
from gym.wrappers import Monitor
from torch.utils.tensorboard import SummaryWriter
from agent import PDQNAgent, RandomAgent
from utils import pad_action

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_dir', default='results/soccer', type=str)
    parser.add_argument('--max_steps', default=5000, type=int)
    parser.add_argument('--train_eps', default=10000, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--epsilon_start', default=0.95, type=float)
    parser.add_argument('--epsilon_decay', default=5000, type=int)
    parser.add_argument('--epsilon_end', default=0.02, type=float)
    parser.add_argument('batch_size', default=32, type=int)
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('param_net_lr', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--layer_actor', default=[256, 128, 64])
    parser.add_argument('--layer_param', default=[256, 128, 64])

    config = parser.parse_args()
    return config


def train(cfg):
    env = gym.make('SoccerScoreGoal-v0')
    env = Monitor(env, directory=os.path.join(cfg.save_dir), video_callable=False, write_upon_reset=False, force=True)
    agent = PDQNAgent(state_space=env.observation_space, action_space=env.action_space,
                      epsilon_start=cfg.epsilon_start, epsilon_decay=cfg.epsilon_decay, epsilon_end=cfg.epsilon_end,
                      batch_size=cfg.batch_size, device=cfg.device, gamma=cfg.gamma,
                      actor_kwargs={"hidden_layers": cfg.layers_actor},
                      param_net_kwargs={"hidden_layers": cfg.layers_param},
                      )

    rewards = []
    moving_avg_rewards = []
    eps_steps = []
    log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_eps in range(1, 1 + cfg.train_eps):
        state = env.reset()
        state = np.array(state, dtype=np.float32)

        episode_reward = 0.
        for i_step in range(cfg.max_steps):
            act, act_param, all_action_param = agent.choose_action(state)
            action = pad_action(act, act_param)
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            agent.memory.push(
                [state, np.concatenate(([act], all_action_param.data)).ravel(), reward, next_state, done])
            episode_reward += reward
            state = next_state
            agent.update()
            if done:
                break
        rewards.append(episode_reward)
        eps_steps.append(i_step)
        if i_eps == 1:
            moving_avg_rewards.append(episode_reward)
        else:
            moving_avg_rewards.append(episode_reward * 0.1 + moving_avg_rewards[-1] * 0.9)
        writer.add_scalars('rewards', {'raw': rewards[-1], 'moving_average': moving_avg_rewards[-1]}, i_eps)
        writer.add_scalar('steps_of_each_trials', eps_steps[-1], i_eps)

        if i_eps % 100 == 0:
            print('Episode: ', i_eps, 'R100: ', moving_avg_rewards[-1], 'n_steps: ', np.array(eps_steps[-100]).mean())

    writer.close()


def evaluation(cfg, saved_model):
    pass


if __name__ == '__main__':
    cfg = get_args()
    if cfg.train:
        train(cfg)
        # evaluation(cfg)
    else:
        pass
        # evaluation(cfg, saved_model=model_path)
