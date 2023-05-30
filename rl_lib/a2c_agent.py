import numpy as np
import random
from rl_lib.a2c_model import ActorCritic, Actor, Critic
import torch
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, nr_agents: int, seed: int,
                 gamma: float = 0.99, learning_rate=1e-4, entropy_weight: float = 1e-5,
                 use_same_network_for_actor_critic: bool = False, constant_var: bool = False):
        """
        @param state_size: dimension of each state
        @param action_size: dimension of each action
        @param nr_agents: number of agents in environment
        @param seed:  random seed
        @param gamma: discount factor
        @param learning_rate: learning rate
        @param entropy_weight: weight for entropy loss
        @param use_same_network_for_actor_critic: if yes, the actor and critic share layers in the NN
        @param constant_var: if no, variance of Gaussian distribution for action sampling is learned
        """

        self.state_size = state_size
        self.action_size = action_size
        self.nr_agents = nr_agents
        self.seed = random.seed(seed)
        self.entropy_weight = entropy_weight

        self.gamma = gamma

        self.learning_rate = learning_rate

        self.use_same_network_for_actor_critic = use_same_network_for_actor_critic

        if self.use_same_network_for_actor_critic:
            self.actor_critic = ActorCritic(state_size, action_size, seed, constant_var).to(device)
            # self.actor_critic_optimizer = optim.RMSprop(self.actor_critic.parameters(), lr=learning_rate, alpha=0.99)
            self.actor_critic_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        else:
            self.actor = Actor(state_size, action_size, seed, constant_var).to(device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

            self.critic = Critic(state_size, seed).to(device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        if self.use_same_network_for_actor_critic:
            self.actor_critic.eval()
            action, action_log_prob, entropy = self.actor_critic.act(state)
            self.actor_critic.train()
        else:
            self.actor.eval()
            action, action_log_prob, entropy = self.actor.act(state)
            self.actor.train()
        return action.cpu().data.numpy(), action_log_prob, entropy

    def save_networks(self, filename):
        if self.use_same_network_for_actor_critic:
            torch.save(self.actor_critic.state_dict(), filename + 'actor_critic.pth')
        else:
            torch.save(self.actor.state_dict(), filename + 'actor.pth')
            torch.save(self.critic.state_dict(), filename + 'critic.pth')

    def learn(self, batch_states, next_states, batch_actions_log_prob, batch_actions_entropy, batch_rewards,
              batch_dones):
        if self.use_same_network_for_actor_critic:
            self._learn_combined(batch_states, next_states, batch_actions_log_prob, batch_actions_entropy,
                                 batch_rewards, batch_dones)
        else:
            self._learn_separate(batch_states, next_states, batch_actions_log_prob, batch_actions_entropy,
                                 batch_rewards, batch_dones)

    def _learn_combined(self, batch_states, next_states, batch_actions_log_prob, batch_actions_entropy, batch_rewards,
                        batch_dones):
        self.actor_critic.eval()

        batch_size = len(batch_dones)
        returns = []
        values = []

        next_states_torch = self._to_torch(next_states)
        final_value = self.actor_critic.value(next_states_torch)
        final_return = final_value * (1 - self._to_torch(batch_dones[-1]))
        for i in reversed(range(batch_size)):
            final_return = self._to_torch(batch_rewards[i]) + self.gamma * final_return
            returns.append(final_return)
            states = self._to_torch(batch_states[i])
            values.append(self.actor_critic.value(states))

        returns.reverse()
        values.reverse()
        self.actor_critic.train()

        values_torch = torch.stack(values).squeeze()
        returns_torch = torch.stack(returns).squeeze()

        advantages = returns_torch - values_torch

        batch_actions_log_prob_torch = torch.stack(batch_actions_log_prob)
        batch_actions_entropy = torch.stack(batch_actions_entropy)

        actor_loss = -(batch_actions_log_prob_torch * advantages.detach()).mean() \
                     - self.entropy_weight * batch_actions_entropy.mean()
        critic_loss = advantages.pow(2).mean()

        total_loss = actor_loss + critic_loss

        self.actor_critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1)
        self.actor_critic_optimizer.step()

    def _learn_separate(self, batch_states, next_states, batch_actions_log_prob, batch_actions_entropy, batch_rewards,
                        batch_dones):
        self.actor.eval()
        self.critic.eval()

        batch_size = len(batch_dones)
        returns = []
        values = []

        next_states_torch = self._to_torch(next_states)
        final_value = self.critic.value(next_states_torch)
        final_return = final_value * (1 - self._to_torch(batch_dones[-1]))
        for i in reversed(range(batch_size)):
            final_return = self._to_torch(batch_rewards[i]) + self.gamma * final_return
            returns.append(final_return)
            states = self._to_torch(batch_states[i])
            values.append(self.critic.value(states))

        returns.reverse()
        values.reverse()
        self.actor.train()
        self.critic.train()

        values_torch = torch.stack(values).squeeze()
        returns_torch = torch.stack(returns).squeeze()

        advantages = returns_torch - values_torch

        batch_actions_log_prob_torch = torch.stack(batch_actions_log_prob)
        batch_actions_entropy = torch.stack(batch_actions_entropy)

        actor_loss = -(batch_actions_log_prob_torch * advantages.detach()).mean() \
                     - self.entropy_weight * batch_actions_entropy.mean()
        critic_loss = advantages.pow(2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

    def _to_torch(self, array):
        return torch.from_numpy(array).float().to(device)
