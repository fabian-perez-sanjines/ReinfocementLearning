
import numpy as np
import gym
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.distributions import Categorical
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPISODES = 100000
GAMMA = 0.99
LEARNING_RATE = 0.0005
NUM_NEURONS = 128

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Critic(nn.Module):
    def __init__(self, n_states):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(n_states, NUM_NEURONS)
        self.value_head = nn.Linear(NUM_NEURONS, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        return state_values


class Actor(nn.Module):
    def __init__(self, n_actions, n_states):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(n_states, NUM_NEURONS)
        self.action_head = nn.Linear(NUM_NEURONS, n_actions)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=-1)


class ActorCritic:

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.learning_step = 0
        self.policy_model = Actor(n_actions, n_states).to(device)
        self.state_value_model = Critic(n_states).to(device)
        self.opt_policy = optim.Adam(
            self.policy_model.parameters(), lr=LEARNING_RATE)
        self.opt_state_value = optim.Adam(
            self.state_value_model.parameters(), lr=LEARNING_RATE)
        self.eps = np.finfo(np.float32).eps.item()

    def update_target_weights(self):
        self.target_action_value = copy.deepcopy(self.action_value)
        self.target_action_value.eval()

    def addReward(self, reward):
        self.state_value_model.rewards.append(reward)

    def learn(self):
        R = 0
        saved_actions = self.state_value_model.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.state_value_model.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        self.opt_policy.zero_grad()
        self.opt_state_value.zero_grad()
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()
        loss.backward()

        self.opt_policy.step()
        self.opt_state_value.step()
        del self.state_value_model.rewards[:]
        del self.state_value_model.saved_actions[:]

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)

        probs = self.policy_model(state)
        m = Categorical(probs)
        action = m.sample()
        state_value = self.state_value_model(state)
        self.state_value_model.saved_actions.append(
            SavedAction(m.log_prob(action), state_value))
        return action.item()


def main():
    print("Start Actor Critic")
    env = gym.make("CartPole-v0")
    n_actions = env.action_space.n
    n_states = np.prod(np.array(env.observation_space.shape))
    print(datetime.today().strftime('%Y-%m-%d-%H:%M'))
    actorCritic = ActorCritic(n_actions, n_states)
    average_cumulative_reward = 0.0
    writer = SummaryWriter('./log_files/')
    writer_name = "Actor_Critic-" + datetime.today().strftime('%Y-%m-%d-%H:%M')
    for episode in range(EPISODES):
        state = env.reset()
        cumulative_reward = 0.0
        steps = 0
        while True:
            steps += 1

            # render
            env.render()

            # act
            action = actorCritic.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            actorCritic.addReward(reward)

            state = next_state
            cumulative_reward += reward

            if done:
                break

        actorCritic.learn()
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
