import numpy as np


class TPERBuffer:
    def __init__(self, state_dim, max_size, batch_size, alpha=0.6,
                 beta=0.4, lam=0.001, seed=None):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.rng = np.random.default_rng(seed)

        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros(max_size, dtype=np.int64)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.bool_)

        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.timestamps = np.zeros(max_size, dtype=np.int64)

    def store_transition(self, state, action, reward, state_, done, timestamp=None):
        idx = self.ptr % self.max_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = state_
        self.terminal_memory[idx] = done

        self.timestamps[idx] = self.ptr if timestamp is None else timestamp
        current_max = self.priorities[:self.size].max() if self.size else 1.0
        self.priorities[idx] = max(current_max, 1e-6)

        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample_buffer(self, current_timestamp=None):
        if current_timestamp is None:
            current_timestamp = self.ptr
        delta_t = np.maximum(0, current_timestamp - self.timestamps[:self.size])
        time_decay = np.exp(-self.lam * delta_t)
        combined_priorities = (self.priorities[:self.size] * time_decay) ** self.alpha
        probs = combined_priorities / np.maximum(combined_priorities.sum(), 1e-12)
        indices = self.rng.choice(
            self.size, self.batch_size, p=probs, replace=False
        )

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
        return self.size >= self.batch_size
