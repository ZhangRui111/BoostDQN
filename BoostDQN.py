import torch
import torch.nn as nn
import numpy as np


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001  # learning rate
EPSILON_INIT = 0
EPSILON_INCRE = 0.001
EPSILON_T = 0.95
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 500
N_ACTIONS = 4
N_STATES = 64


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
        )
        self.out = nn.Linear(50, N_ACTIONS)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        actions_value = self.out(x)
        return actions_value


class BoostPrior(object):
    def __init__(self, path, to_tensor=True):
        super().__init__()
        self.path = path
        self.prior_states, self.target_actions = self.load_priors()
        if self.target_actions.ndim == 1:
            self.target_actions = self.target_actions.reshape(self.prior_states.shape[0], 1)
        if to_tensor:
            self.prior_states = torch.from_numpy(self.prior_states).float().to('cuda')
            self.target_actions = torch.from_numpy(self.target_actions).float().to('cuda')

    def load_priors(self):
        return np.loadtxt("{}states_prior.npy".format(self.path)), \
               np.loadtxt("{}actions_target.npy".format(self.path))


class BoostDQN(object):
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.eval_net, self.target_net = Net().to('cuda'), Net().to('cuda')

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory_capacity = MEMORY_CAPACITY
        self.epsilon = EPSILON_INIT
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_dqn_func = nn.MSELoss()
        self.loss_prior_func = nn.MSELoss()

        self.prior_beta = 1
        self.prior = BoostPrior("./data/map2/")

    def choose_action(self, x):
        x = torch.from_numpy(x).float().to('cuda')
        x = x.unsqueeze(0)
        if np.random.uniform() < self.epsilon:
            # print("Greedy action selection")
            actions_value = self.eval_net(x)
            _, action = torch.max(actions_value, 1)
        else:
            # print("Random action selection")
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :N_STATES]
        b_a = b_memory[:, N_STATES:N_STATES+1].astype(int)
        b_r = b_memory[:, N_STATES+1:N_STATES+2]
        b_s_ = b_memory[:, -N_STATES:]
        b_s = torch.from_numpy(b_s).float().to('cuda')
        b_a = torch.from_numpy(b_a).float().to('cuda')
        b_r = torch.from_numpy(b_r).float().to('cuda')
        b_s_ = torch.from_numpy(b_s_).float().to('cuda')

        # normal dqn loss
        q_eval = self.eval_net(b_s).gather(1, b_a.long())  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach().max(dim=1)[0]  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.view(self.batch_size, 1)  # shape (batch, 1)
        loss_dqn = self.loss_dqn_func(q_eval, q_target)
        # prior loss
        prior_q_eval = self.eval_net(self.prior.prior_states)
        prior_q_eval_max, _ = prior_q_eval.max(dim=1)
        prior_q_target = prior_q_eval.gather(1, self.prior.target_actions.long())
        loss_prior = self.loss_prior_func(prior_q_eval_max, prior_q_target)
        # total loss
        loss = loss_dqn + self.prior_beta * loss_prior

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon += EPSILON_INCRE
        self.epsilon = min(self.epsilon, EPSILON_T)
