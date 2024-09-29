import os
import torch
import numpy as np
import random
from preprocess import DataReader
import datetime as dt
import matplotlib.pyplot as plt
from dqn import DQN
from trading_env import TradingSystem_v0

curr_path = os.path.dirname(__file__)
curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
random.seed(11)


class Config:

    def __init__(self):
        ################################## env hyperparameters ###################################
        self.env_name = 'TradingSystem_v0' # environment name
        self.device = "cpu"
        self.seed = 11 # random seed
        self.train_eps = 200 # training episodes
        self.state_space_dim = 50 # state space size (K-value)
        self.action_space_dim = 3 # action space size (short: 0, neutral: 1, long: 2)
        ################################################################################

        ################################## algo hyperparameters ###################################
        self.gamma = 0.95  # discount factor
        self.epsilon_start = 0.90  # start epsilon of e-greedy policy
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 300  # capacity of experience replay
        self.batch_size = 32  # size of mini-batch SGD
        self.target_update = 4  # update frequency of target network
        self.hidden_dim = 128  # dimension of hidden layer
        ################################################################################

        ################################# save path ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        self.save = True  # whether to save the image
        ################################################################################


def env_agent_config(data, cfg):
    env = TradingSystem_v0(data, cfg.state_space_dim)
    agent = DQN(cfg.state_space_dim * data.columns.shape[0], cfg.action_space_dim, cfg)
    if cfg.seed != 0:  # set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent


def train(cfg, env, agent):
    print('Start Training!')
    print(f'Environment：{cfg.env_name}')
    rewards = []  # record total rewards
    ma_rewards = []  # record moving average total rewards
    for i_ep in range(cfg.train_eps):
        ep_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state, 1)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('Episode：{}/{}, Reward：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('Finish Training!')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    rewards = []  # record total rewards
    ep_reward = 0
    state = env.reset()
    while True:
        action = agent.choose_action(state, 0)
        next_state, reward, done = env.step(action)
        state = next_state
        ep_reward += reward
        action_dict = {0 : 'SHORT',
                       1 : 'HOLD',
                       2 :  'BUY'}
        print(f"Action : {action_dict[action]}")
        print(f"Reward : {reward}")
        print(f"Episode reward : {ep_reward}")
        if done:
            break
    rewards.append(ep_reward)
    print(f"Episode：1/1，Reward：{ep_reward:.1f}")
    print('Finish Testing!')
    return rewards

if __name__ == "__main__":

    symbol = 'THYAO'
    period = '5'
    at = '20240831'
    last = '1440'
    # csv algoları eklenecek
    reader = DataReader(symbol, at, last, period)
    reader.read_data()
    reader.extract_indicators()
    train_data, test_data = reader.split_data()
    train_data, test_data = train_data[['pct_change']], test_data[['pct_change']]
    cfg = Config()

    # training
    env, agent = env_agent_config(train_data, cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    os.makedirs(cfg.result_path)  # create output folders
    os.makedirs(cfg.model_path)
    agent.save(path=cfg.model_path)  # save model
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))   # plot the training result
    ax.plot(list(range(1, cfg.train_eps+1)), rewards, color='blue', label='rewards')
    ax.plot(list(range(1, cfg.train_eps+1)), ma_rewards, color='green', label='ma_rewards')
    ax.legend()
    ax.set_xlabel('Episode')
    plt.savefig(cfg.result_path+'train.jpg')

    # testing
    env, agent = env_agent_config(test_data, cfg)
    agent.load(path=cfg.model_path)  # load model
    rewards = test(cfg, env, agent)
    buy_and_hold_rewards = test_data['close']



