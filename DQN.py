import torch
import torch.nn as nn
import numpy as np

from utils.utils import parse_model_config


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_states, 50),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
        )
        self.out = nn.Linear(50, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, params_path):
        self.info = 'dqn'
        self.params = parse_model_config(params_path)
        self.batch_size = int(self.params['batch_size'])
        self.lr = float(self.params['lr'])
        self.epsilon = float(self.params['epsilon_init'])
        self.epsilon_incre = float(self.params['epsilon_incre'])
        self.epsilon_target = float(self.params['epsilon_target'])
        self.gamma = float(self.params['gamma'])
        self.target_replace_iter = int(self.params['target_replace_iter'])
        self.memory_capacity = int(self.params['memory_capacity'])
        self.n_actions = int(self.params['n_actions'])
        self.n_states = int(self.params['n_states'])

        self.eval_net = Net(self.n_states, self.n_actions).to('cuda')
        self.target_net = Net(self.n_states, self.n_actions).to('cuda')

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.memory_capacity, self.n_states * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.from_numpy(x).float().to('cuda').unsqueeze(0)
        if np.random.uniform() < self.epsilon:
            # print("Greedy action selection")
            actions_value = self.eval_net(x)
            _, action = torch.max(actions_value, 1)
        else:
            # print("Random action selection")
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = b_memory[:, :self.n_states]
        b_a = b_memory[:, self.n_states:self.n_states+1].astype(int)
        b_r = b_memory[:, self.n_states+1:self.n_states+2]
        b_s_ = b_memory[:, -self.n_states:]
        b_s = torch.from_numpy(b_s).float().to('cuda')
        b_a = torch.from_numpy(b_a).float().to('cuda')
        b_r = torch.from_numpy(b_r).float().to('cuda')
        b_s_ = torch.from_numpy(b_s_).float().to('cuda')

        # q_eval w.r.t the action in experience
        q_eval_s = self.eval_net(b_s)
        q_eval = q_eval_s.gather(1, b_a.long())  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach().max(1)[0]  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_epsilon(self):
        self.epsilon += self.epsilon_incre
        self.epsilon = min(self.epsilon, self.epsilon_target)
