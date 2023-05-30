import torch
import torch.nn as nn
import torch.nn.functional
import torch.distributions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, constant_var=True, fc1_units=64, fc2_units=64):
        """
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            constant_var (bool): if false, variance of Gaussian distributions is learned, otherwise it is kept constant
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.constant_var = constant_var
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # actor output layers
        self.mean = nn.Linear(fc2_units, action_size)
        self.var = nn.Linear(fc2_units, action_size)
        self.soft_plus = torch.nn.Softplus(beta=1, threshold=1)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return x

    def act(self, state):
        x = self.forward(state)

        mean = self.mean(x)

        if self.constant_var:
            std = torch.ones(self.action_size).to(device)
        else:
            std = torch.sqrt(self.soft_plus(self.var(x))) + 1e-6

        distribution = torch.distributions.Normal(mean, std)
        samples = distribution.sample()

        entropy = torch.sum(distribution.entropy(), dim=1, keepdim=False)
        actions_log_prob = torch.sum(distribution.log_prob(samples), dim=1, keepdim=False)
        actions = torch.tanh(samples)
        return actions, actions_log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_size, seed, fc1_units=64, fc2_units=64):
        """
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # critic output layer
        self.critic = nn.Linear(fc2_units, 1)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return x

    def value(self, state):
        x = self.forward(state)
        return self.critic(x).squeeze(1)


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, seed, constant_var=True, fc1_units=128, fc2_units=64):
        """ Shared actor/critic network inspired by https://arxiv.org/pdf/1602.01783.pdf
        Similar to https://arxiv.org/pdf/1602.01783.pdf for experiments with discrete action domains,
        all non-output layer are shared between the actor and the critic.

        We use a normal distribution to sample actions.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            constant_var (bool): if false, variance of Gaussian distributions is learned, otherwise it is kept constant
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.constant_var = constant_var
        self.state_size = state_size
        self.action_size = action_size

        # shared layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # actor output layers
        self.actor_mean = nn.Linear(fc2_units, action_size)
        self.var = nn.Linear(fc2_units, action_size)
        self.soft_plus = torch.nn.Softplus(beta=1, threshold=1)

        # critic output layer
        self.critic = nn.Linear(fc2_units, 1)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return x

    def act(self, state):
        x = self.forward(state)

        mean = self.actor_mean(x)

        if self.constant_var:
            std = torch.ones(self.action_size).to(device)
        else:
            std = torch.sqrt(self.soft_plus(self.var(x))) + 1e-6

        distribution = torch.distributions.Normal(mean, std)

        samples = distribution.sample()

        entropy = torch.sum(distribution.entropy(), dim=1, keepdim=False)
        actions_log_prob = torch.sum(distribution.log_prob(samples), dim=1, keepdim=False)
        actions = torch.tanh(samples)
        return actions, actions_log_prob, entropy

    def value(self, state):
        x = self.forward(state)
        return self.critic(x).squeeze(1)
