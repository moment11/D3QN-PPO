import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from buffer import TPERBuffer

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)

        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.0)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        Q = V + (A - T.mean(A, dim=-1, keepdim=True))
        return Q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file, map_location=device, weights_only=True))


class D3QN:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir,
                 gamma=0.99, epsilon=1.0, eps_end=0.01,
                 max_size=100000, batch_size=256,
                 tper_alpha=0.6, tper_beta=0.4, tper_lam=0.001,
                 target_update_interval=1000, seed=None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.action_space = list(range(action_dim))
        self.target_update_interval = target_update_interval
        self.learn_step = 0
        self.rng = np.random.default_rng(seed)

        self.q_eval = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                          fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_target = DuelingDeepQNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=fc1_dim, fc2_dim=fc2_dim)


        self.memory = TPERBuffer(state_dim=state_dim, max_size=max_size,
                                 batch_size=batch_size, alpha=tper_alpha,
                                 beta=tper_beta, lam=tper_lam, seed=seed)

        self.update_target_network()
        self.eps_decay_factor = 0.999

    def update_target_network(self):
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_decay_factor \
            if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, is_train=True):
        state = T.tensor(np.array([observation]), dtype=T.float).to(device)
        q_vals = self.q_eval.forward(state)
        action = T.argmax(q_vals).item()

        if self.rng.random() < self.epsilon and is_train:
            action = int(self.rng.choice(self.action_space))
        return action

    def learn(self):
        if not self.memory.ready():
            return None


        states, actions, rewards, next_states, terminals, indices, weights = self.memory.sample_buffer()

        batch_idx = T.arange(self.batch_size, dtype=T.long).to(device)
        states_tensor = T.tensor(states, dtype=T.float).to(device)
        actions_tensor = T.tensor(actions, dtype=T.long).to(device)
        rewards_tensor = T.tensor(rewards, dtype=T.float).to(device)
        next_states_tensor = T.tensor(next_states, dtype=T.float).to(device)
        terminals_tensor = T.tensor(terminals).to(device)
        weights_tensor = T.tensor(weights, dtype=T.float).to(device)


        q_eval = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]


        with T.no_grad():
            max_actions = T.argmax(self.q_eval.forward(next_states_tensor), dim=-1)
            q_next = self.q_target.forward(next_states_tensor)[batch_idx, max_actions]
            q_next[terminals_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_next

        td_errors = T.abs(target - q_eval).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)


        loss = (weights_tensor * (q_eval - target) ** 2).mean()

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_interval == 0:
            self.update_target_network()
        self.decrement_epsilon()
        return {"d3qn_loss": float(loss.item()), "mean_td_error": float(td_errors.mean())}

    def save_models(self, episode):
        for sub in ['Q_eval', 'Q_target']:
            path = os.path.join(self.checkpoint_dir, sub)
            if not os.path.exists(path):
                os.makedirs(path)

        self.q_eval.save_checkpoint(os.path.join(self.checkpoint_dir, 'Q_eval/D3QN_q_eval_{}.pth'.format(episode)))
        self.q_target.save_checkpoint(
            os.path.join(self.checkpoint_dir, 'Q_target/D3QN_Q_target_{}.pth'.format(episode)))

    def load_models(self, episode):
        self.q_eval.load_checkpoint(os.path.join(self.checkpoint_dir, 'Q_eval/D3QN_q_eval_{}.pth'.format(episode)))
        self.q_target.load_checkpoint(
            os.path.join(self.checkpoint_dir, 'Q_target/D3QN_Q_target_{}.pth'.format(episode)))
