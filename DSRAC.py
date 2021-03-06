import torch.nn as nn
import torch.optim as optim
import torch.autograd
import numpy as np
import random
from memory import memory
from tensorboardX import SummaryWriter
from datetime import datetime
import copy
import torch.nn.functional as F
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 800
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON_END = 0.1
DECAY_EPSILON = 0.99
MEMORY_SIZE = 10000
NUM_FEATURES = 64
# each element in the list is one of the intermediate layers with that number
# of neurons, the mu model always add a first and end last layers of size of
# NUM_FEATURES
LAYERS = [8]
REPLACE_TARGET_FREQUENCY = 20
BATCH_SIZE = 16


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, output_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Feature(nn.Module):
    def __init__(self, n_states):
        super(Feature, self).__init__()
        self.hidden = nn.Linear(n_states, NUM_FEATURES)

    def forward(self, states):
        features = F.relu(self.hidden(states))
        return features


class Reward(nn.Module):
    def __init__(self, features_layer, q_reward_layer):
        super(Reward, self).__init__()
        self.q_reward_layer = q_reward_layer
        self.features_layer = features_layer

    def forward(self, states):
        features = self.features_layer(states)
        reward = self.q_reward_layer(features)
        return reward


class QReward(nn.Module):
    def __init__(self):
        super(QReward, self).__init__()
        self.reward = nn.Linear(NUM_FEATURES, 1)

    def forward(self, mu):
        reward = self.reward(mu)
        return reward


class Critic(nn.Module):
    def __init__(self, num_actions, features_layer):
        super(Critic, self).__init__()
        self.num_actions = num_actions
        self.features_layer = features_layer
        self.num_layers = len(LAYERS)
        for i in range(num_actions):
            setattr(self, "layer_%d_action_%d_" %
                    (1, i), nn.Linear(NUM_FEATURES + num_actions, LAYERS[0]))
            for j in range(self.num_layers):
                layer = LAYERS[j]
                if j >= self.num_layers - 1:
                    next_layer = NUM_FEATURES
                else:
                    next_layer = LAYERS[j + 1]

                setattr(self, "layer_%d_action_%d_" %
                        (j + 2, i), nn.Linear(layer, next_layer))
            setattr(self, "layer_%d_action_%d_" %
                    (self.num_layers + 2, i), nn.Linear(NUM_FEATURES, NUM_FEATURES))

    def forward(self, states, actions):
        features = self.features_layer(states).detach()
        x = torch.cat([features, actions], 1)
        mu = torch.zeros([self.num_actions, states.size()[0],
                          NUM_FEATURES], dtype=torch.float)
        for i in range(self.num_actions):
            layer_result = x
            for j in range(self.num_layers + 1):
                layer = getattr(self, "layer_%d_action_%d_" % (j + 1, i))
                layer_result = F.relu(layer(layer_result))

            layer = getattr(self, "layer_%d_action_%d_" %
                            (self.num_layers + 2, i))
            successor_represantation_i = layer(layer_result)
            mu[i] = successor_represantation_i

        return mu


class Q(nn.Module):
    def __init__(self, num_actions, mu, q_reward_layer):
        super(Q, self).__init__()
        self.num_actions = num_actions
        self.mu = mu
        self.q_reward_layer = q_reward_layer

    def forward(self, states, actions):
        mu = F.relu(self.mu(states, actions))
        q = self.q_reward_layer(mu)
        return q


class DSR:
    def __init__(self, n_actions, n_states):
        print(n_states)
        self.n_actions = n_actions
        self.n_states = n_states
        self.learning_step = 0
        self.feature = Feature(n_states).to(device)
        self.qReward = QReward().to(device)
        self.reward = Reward(self.feature, self.qReward).to(device)
        self.mu = Critic(n_actions, self.feature).to(device)
        self.mu_target = Critic(n_actions, self.feature).to(device)
        self.q = Q(n_actions, self.mu, self.qReward).to(device)
        self.q_target = Q(n_actions, self.mu_target,
                          self.qReward).to(device)

        self.actor = Actor(self.n_states, self.n_actions)
        self.actor_target = Actor(
            self.n_states, self.n_actions)

        self.critic_optimizer = optim.Adam(
            self.mu.parameters(), lr=LEARNING_RATE)
        self.opt_reward = optim.Adam(
            self.reward.parameters(), lr=LEARNING_RATE)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    def update_target_weights(self):
        self.mu_target = copy.deepcopy(self.mu)
        self.q_target = Q(
            self.n_actions, self.mu_target, self.qReward).to(device)
        self.mu_target.eval()

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action_scores = self.actor_target.forward(state)
        return action_scores

    def learn(self, states, actions, rewards, next_states, dones):
        if self.learning_step % REPLACE_TARGET_FREQUENCY == 0:
            self.update_target_weights()
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)
        not_dones = (1 - dones).unsqueeze(1)

        rewards = rewards.unsqueeze(1)
        # estimate rewards
        expected_rewards = self.reward(states)

        loss_r = self.loss_func(expected_rewards, rewards)

        # Critic loss
        next_actions = self.actor_target(next_states).detach()
        qs_next_state = self.q_target(next_states, next_actions).detach()
        qs_next_state = qs_next_state.squeeze()
        qs_next_state = qs_next_state.permute(1, 0)

        expected_mu_next_state = self.mu_target(
            next_states, qs_next_state).detach()

        features = self.feature(states).detach()

        target_mu = (features + (GAMMA * expected_mu_next_state * (not_dones)))
        expected_mu = self.mu(states, actions)
        loss_mu = self.loss_func(expected_mu, target_mu)

        # Actor loss
        actions_actor = self.actor(states)
        log_actions = torch.log(actions_actor)
        advantage = self.q_target(states, actions_actor).detach()
        policy_loss = -(log_actions * advantage).mean()

        # update networks

        self.opt_reward.zero_grad()
        loss_r.backward()
        self.opt_reward.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        loss_mu.backward()
        self.critic_optimizer.step()


def main():
    print("Start dsrac")
    env = gym.make("CartPole-v0")
    memories = memory(MEMORY_SIZE)
    n_actions = env.action_space.n
    n_states = np.prod(np.array(env.observation_space.shape))
    print(datetime.today().strftime('%Y-%m-%d-%H:%M'))
    dsrac = DSR(n_actions, n_states)
    average_cumulative_reward = 0.0
    writer = SummaryWriter('./log_files/')
    writer_name = "DSRAC-" + datetime.today().strftime('%Y-%m-%d-%H:%M')
    epsilon = 1
    for episode in range(EPISODES):
        state = env.reset()
        cumulative_reward = 0.0
        steps = 0
        while True:
            steps += 1
            dsrac.learning_step += 1

            # render
            env.render()

            # act
            action_scores = dsrac.choose_action(state).squeeze()

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
                dsrac.learn(*memories.sample(batch))

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
