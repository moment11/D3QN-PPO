import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class EdgeComputingEnv(gym.Env):
    """Cloud-edge-end IIoT environment described in the paper."""

    TASK_TYPE_WEIGHTS = np.asarray(
        [[0.45, 0.45, 0.10], [0.40, 0.30, 0.30], [0.25, 0.15, 0.60]],
        dtype=np.float32,
    )

    def __init__(self, num_devices=30, num_edges=5, max_task_size=10.0,
                 max_deadline=3.0, max_steps_per_episode=200, seed=None,
                 priority_weights=None):
        super().__init__()
        self.num_devices = num_devices
        self.num_edges = num_edges
        self.max_task_size = max_task_size
        self.max_deadline = max_deadline
        self.max_steps_per_episode = max_steps_per_episode
        self.rng = np.random.default_rng(seed)
        self.priority_weight_matrix = (
            self.TASK_TYPE_WEIGHTS.copy()
            if priority_weights is None
            else np.tile(np.asarray(priority_weights, dtype=np.float32), (3, 1))
        )
        if not np.allclose(self.priority_weight_matrix.sum(axis=1), 1.0):
            raise ValueError("Each priority weight vector must sum to one")

        self.time_step_duration = 0.1
        self.bandwidth_hz = 20e6
        self.noise_watt = 10 ** ((-100.0 - 30.0) / 10.0)
        self.tx_power_watt = 0.2
        self.backhaul_power_watt = 1.0
        self.backhaul_rate_bps = 1e9
        self.cloud_cpu_hz = 100e9
        self.energy_coefficient = 1e-27
        self.completion_reward = 50.0
        self.base_overdue_penalty = 20.0
        self.priority_weight = 10.0
        self.delay_weight = 1.5
        self.energy_weight = 0.8
        self.load_balance_weight = 0.05

        self.edge_cpu_hz = self.rng.uniform(20e9, 30e9, size=num_edges)
        self.edge_power_watt = self.rng.uniform(0.1, 0.2, size=num_edges)
        self.server_loads = np.zeros(num_edges, dtype=np.float64)
        self.channel_gains = np.ones(num_edges, dtype=np.float64)
        self.task_queue = deque()
        self.current_task = None
        self.episode_step = 0

        # Five task/device values plus capacity, load, and channel per MEC.
        self.state_dim = 5 + 3 * self.num_edges
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        # 0..M-1 select a MEC; M selects cloud through the least-loaded MEC.
        self.action_space = spaces.Dict({
            "location": spaces.Discrete(self.num_edges + 1),
            "continuous_params": spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            ),
        })

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_step = 0
        self.server_loads.fill(0.0)
        self.task_queue.clear()
        self._get_next_task()
        self._refresh_channel_gains()
        return self._get_state()

    def _generate_tasks(self, num_tasks):
        for _ in range(num_tasks):
            task_type = int(self.rng.integers(0, len(self.priority_weight_matrix)))
            deadline = self.rng.uniform(0.5, min(2.0, self.max_deadline))
            factors = np.asarray([
                np.clip(1.0 - deadline / self.max_deadline, 0.0, 1.0),
                self.rng.uniform(0.0, 1.0),
                self.rng.uniform(0.0, 1.0),
            ])
            priority = float(np.dot(self.priority_weight_matrix[task_type], factors))
            self.task_queue.append({
                "size_mb": self.rng.uniform(1.0, self.max_task_size),
                "complexity": self.rng.uniform(500.0, 1500.0),
                "deadline": deadline,
                "task_type": task_type,
                "priority": priority,
                "battery": self.rng.uniform(0.5, 1.0),
                "local_cpu_hz": self.rng.uniform(0.5e9, 1.5e9),
            })

    def _get_next_task(self):
        if not self.task_queue:
            self._generate_tasks(self.num_devices)
        self.current_task = self.task_queue.popleft()

    def _refresh_channel_gains(self):
        distances = self.rng.uniform(10.0, 1000.0, size=self.num_edges)
        unit_gain = 10 ** (-40.0 / 10.0)
        self.channel_gains = unit_gain * distances ** -2

    def _get_state(self):
        task = self.current_task
        normalized_loads = np.clip(
            self.server_loads / (self.edge_cpu_hz * self.time_step_duration),
            0.0,
            1.0,
        )
        normalized_channels = self.channel_gains / (self.channel_gains + 1e-10)
        return np.concatenate([
            [task["size_mb"] / self.max_task_size,
             task["complexity"] / 1500.0,
             task["priority"],
             task["battery"],
             task["local_cpu_hz"] / 1.5e9],
            self.edge_cpu_hz / 30e9,
            normalized_loads,
            normalized_channels,
        ]).astype(np.float32)

    def step(self, action):
        if self.episode_step % self.num_devices == 0:
            self.server_loads = np.maximum(
                0.0,
                self.server_loads - self.edge_cpu_hz * self.time_step_duration,
            )
        location = int(action["location"])
        if not self.action_space["location"].contains(location):
            raise ValueError(f"Invalid location action: {location}")
        alpha, beta = np.clip(
            np.asarray(action["continuous_params"], dtype=np.float64), 0.0, 1.0
        )

        task = self.current_task
        data_bits = task["size_mb"] * 1e6
        data_bytes = data_bits / 8.0
        cycles = data_bytes * task["complexity"]
        local_cycles = (1.0 - alpha) * cycles
        local_delay = local_cycles / task["local_cpu_hz"]
        local_energy = self.energy_coefficient * local_cycles * task["local_cpu_hz"] ** 2

        mec_index = location if location < self.num_edges else int(
            np.argmin(self.server_loads / self.edge_cpu_hz)
        )
        rate = self.bandwidth_hz * np.log2(
            1.0 + self.tx_power_watt * self.channel_gains[mec_index] / self.noise_watt
        )
        radio_delay = alpha * data_bits / max(rate, 1.0)
        radio_energy = self.tx_power_watt * radio_delay
        offloaded_cycles = alpha * cycles

        if location < self.num_edges:
            allocated_cpu = max(beta, 1e-3) * self.edge_cpu_hz[mec_index]
            remote_delay = offloaded_cycles / allocated_cpu
            remote_energy = remote_delay * self.edge_power_watt[mec_index] * beta
            backhaul_delay = 0.0
            self.server_loads[mec_index] += offloaded_cycles
        else:
            backhaul_delay = alpha * data_bits / self.backhaul_rate_bps
            remote_delay = offloaded_cycles / self.cloud_cpu_hz
            remote_energy = self.backhaul_power_watt * backhaul_delay

        total_delay = max(local_delay, radio_delay + backhaul_delay + remote_delay)
        total_energy = local_energy + radio_energy + remote_energy
        deadline = task["deadline"]
        priority = task["priority"]
        success = total_delay <= deadline
        normalized_overdue = max(0.0, total_delay - deadline) / (deadline + 1e-8)
        utilization = self.server_loads / (
            self.edge_cpu_hz * self.time_step_duration + 1e-8
        )
        average_utilization = np.mean(utilization)
        load_imbalance = np.mean(
            ((utilization - average_utilization) / (average_utilization + 1e-8)) ** 2
        )
        reward = (
            float(success) * (self.completion_reward + self.priority_weight * priority)
            - self.delay_weight
            * (self.base_overdue_penalty + self.priority_weight * priority)
            * normalized_overdue
            - self.energy_weight * total_energy
            - self.load_balance_weight * load_imbalance
        )

        self.episode_step += 1
        done = self.episode_step >= self.max_steps_per_episode
        info = {
            "is_success": bool(success),
            "delay_s": float(total_delay),
            "energy_j": float(total_energy),
            "priority": float(priority),
            "task_type": int(task["task_type"]),
            "location": location,
            "alpha": float(alpha),
            "beta": float(beta),
        }
        self._get_next_task()
        if self.episode_step % self.num_devices == 0:
            self._refresh_channel_gains()
        return self._get_state(), float(reward), done, info