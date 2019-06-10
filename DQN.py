
import numpy as np
import random
import gym
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import memory
import copy
from tensorboardX import SummaryWriter
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPISODES = 800
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.0005
EPSILON_END = 0.01
DECAY_EPSILON = 0.995
REPLACE_TARGET_FREQUENCY = 4
RANDOM_SEED = 1110
MEMORY_SIZE = 100000


class Model(nn.Module):
    def __init__(self, n_actions, n_states):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(n_states, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DQN:

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.learning_step = 0
        self.action_value = Model(n_actions, n_states).to(device)
        self.target_action_value = Model(n_actions, n_states).to(device)
        self.update_target_weights()
        self.opt = optim.Adam(self.action_value.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def update_target_weights(self):
        self.target_action_value = copy.deepcopy(self.action_value)
        self.target_action_value.eval()

    def learn(self, states, actions, rewards, states_next, dones):
        self.learning_step += 1
        if self.learning_step % REPLACE_TARGET_FREQUENCY == 0:
            self.update_target_weights()

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        states_next = torch.from_numpy(states_next).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)

        Q_targets_next = self.target_action_value(
            states_next).detach().max(1)[0]

        Q_targets = (rewards + (GAMMA * Q_targets_next * (1 - dones)))
        Q_targets = Q_targets.unsqueeze(1)
        expected_values = self.action_value(
            states).gather(1, actions.unsqueeze(1))
        self.opt.zero_grad()
        loss = F.mse_loss(expected_values, Q_targets)

        loss.backward()
        self.opt.step()

    def choose_action(self, state, epsilon):
        p = np.random.random()
        if p > epsilon:
            state = torch.tensor(state, dtype=torch.float)
            return self.action_value(state).max(0)[1].item()
        else:
            return random.randrange(self.n_actions)


def main():
    print("Start dqn")
    env = gym.make("CartPole-v0")
    env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    memories = memory(MEMORY_SIZE)
    n_actions = env.action_space.n
    n_states = np.prod(np.array(env.observation_space.shape))
    print(datetime.today().strftime('%Y-%m-%d-%H:%M'))
    dqn = DQN(n_actions, n_states)
    average_cumulative_reward = 0.0
    writer = SummaryWriter('./log_files/')
    writer_name = "DQN-" + datetime.today().strftime('%Y-%m-%d-%H:%M')
    epsilon = 1.0
    for episode in range(EPISODES):
        state = env.reset()
        cumulative_reward = 0.0
        steps = 0
        while True:
            steps += 1

            # render
            env.render()

            # act
            action = dqn.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            memories.remember(state, action, reward, next_state, done)
            size = memories.size

            # learn
            if size > BATCH_SIZE * 2:
                batch = random.sample(range(size), BATCH_SIZE)
                dqn.learn(*memories.sample(batch))

            state = next_state
            cumulative_reward += reward

            if done:
                break

        epsilon = max(EPSILON_END, epsilon * DECAY_EPSILON)
        # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward

        print(episode, cumulative_reward, average_cumulative_reward)

        writer.add_scalar("Cumulative Reward " + writer_name,
                          cumulative_reward, episode)
        writer.add_scalar("Mean Cumulative Reward" + writer_name,
                          average_cumulative_reward, episode)

    print('Complete')
    env.render()
    env.close()


if __name__ == '__main__':
    main()
