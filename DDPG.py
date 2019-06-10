
import numpy as np
import gym
import random
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from torch.autograd import Variable
from memory import memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPISODES = 1000
BATCH_SIZE = 16
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
TAU = 1e-2


class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class ActorCritic:

    def __init__(self, num_actions, num_states):
        # Params
        self.gamma = GAMMA
        self.tau = TAU
        self.num_states = num_states
        self.num_actions = num_actions
        # Networks
        self.actor = Actor(self.num_states, self.num_actions)
        self.actor_target = Actor(
            self.num_states, self.num_actions)
        self.critic = Critic(
            self.num_states + self.num_actions, self.num_actions)
        self.critic_target = Critic(
            self.num_states + self.num_actions, self.num_actions)

        self.update_target(self.critic_target, self.critic)
        self.update_target(self.actor_target, self.actor)

        # Training
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=LEARNING_RATE)

    def update_target(self, target, original):
        for target_param, param in zip(target.parameters(),
                                       original.parameters()):
            target_param.data.copy_(
                (1.0 - TAU) * target_param.data + TAU * param.data)

    def learn(self, states, actions, rewards, states_next, dones):

        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).to(device)
        next_states = torch.from_numpy(states_next).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        not_dones = (1 - dones).unsqueeze(1)
        rewards = rewards.unsqueeze(1)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states).detach()
        next_Q = self.critic_target.forward(next_states, next_actions)
        Qprime = (rewards + (self.gamma * next_Q * (not_dones)))
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = - \
            self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_target(self.critic_target, self.critic)
        self.update_target(self.actor_target, self.actor)

    def choose_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action_scores = self.actor.forward(state)
        return action_scores


def main():
    print("Start Actor Critic")
    env = gym.make("CartPole-v0")
    memories = memory(MEMORY_SIZE)
    n_actions = env.action_space.n
    n_states = np.prod(np.array(env.observation_space.shape))
    print(datetime.today().strftime('%Y-%m-%d-%H:%M'))
    ac = ActorCritic(n_actions, n_states)
    average_cumulative_reward = 0.0
    writer = SummaryWriter('./log_files/')
    writer_name = "Actor_Cricitic-Experience_Replay-" + \
        datetime.today().strftime('%Y-%m-%d-%H:%M')
    epsilon = 1
    for episode in range(EPISODES):
        state = env.reset()
        cumulative_reward = 0.0
        steps = 0
        while True:
            steps += 1

            # render
            env.render()

            # act
            action_scores = ac.choose_action(state).squeeze()

            p = np.random.random()
            if p > epsilon:
                action = action_scores.max(0)[1].item()
            else:
                action = random.randrange(n_actions)

            next_state, reward, done, _ = env.step(action)
            action_scores = action_scores.data.numpy()
            memories.remember(state, action_scores, reward, next_state, done)
            size = memories.size

            # learn
            if size > BATCH_SIZE * 2:
                batch = random.sample(range(size), BATCH_SIZE)
                ac.learn(*memories.sample(batch))

            state = next_state
            cumulative_reward += reward

            if done:
                break

        epsilon = epsilon * 0.95
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
