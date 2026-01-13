import numpy as np
import time


class TPERBuffer:
    def __init__(self, state_dim, max_size, batch_size, alpha=0.6, beta=0.4, lam=0.001):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0


        self.alpha = alpha
        self.beta = beta
        self.lam = lam

        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros(max_size, dtype=np.int64)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)

        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.timestamps = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.ptr % self.max_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = state_
        self.terminal_memory[idx] = done

        self.timestamps[idx] = time.time()

        self.priorities[idx] = max(self.priorities.max(), 1.0) if self.size > 0 else 1.0

        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample_buffer(self):
        current_time = time.time()

        delta_t = current_time - self.timestamps[:self.size]
        time_decay = np.exp(-self.lam * delta_t)

        combined_priorities = (self.priorities[:self.size] * time_decay) ** self.alpha


        probs = combined_priorities / combined_priorities.sum()

        indices = np.random.choice(self.size, self.batch_size, p=probs, replace=False)


        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.next_state_memory[indices]
        terminals = self.terminal_memory[indices]

        return states, actions, rewards, states_, terminals, indices, weights

    def update_priorities(self, indices, errors):

        self.priorities[indices] = np.abs(errors) + 1e-6  

    def ready(self):
        return self.size > self.batch_size
