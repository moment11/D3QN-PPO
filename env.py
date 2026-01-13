import gym
from gym import spaces
import numpy as np
from collections import deque

class EdgeComputingEnv(gym.Env):
    def __init__(self, num_devices=30, num_edges=5, max_task_size=10.0, max_deadline=3000,
                 
                 R_base=50.0,
                 P_base=20.0,
                 w_tau=10.0,
                 w_O=1.5,
                 w_E=0.8,
                 w_L=5.0,
                 C_local=1.0,
                 B_up=20.0,
                 w_realtime=0.4, w_safety=0.4, w_data=0.2):
        super(EdgeComputingEnv, self).__init__()

        self.num_devices = num_devices
        self.num_edges = num_edges
        self.max_task_size = max_task_size
        self.max_deadline = max_deadline
        self.C_local = C_local

        
        self.R_base = R_base
        self.P_base = P_base
        self.w_tau = w_tau
        self.w_O = w_O
        self.w_E = w_E
        self.w_L = w_L

        
        self.C_edges = np.random.uniform(20.0, 30.0, size=num_edges)  
        self.e_edge_p = np.random.uniform(0.1, 0.2, size=num_edges)  
        self.B_up = B_up
        self.e_upload = 0.05
        self.time_step_duration = 0.1 

        self.w_realtime = w_realtime
        self.w_safety = w_safety
        self.w_data = w_data
        self.max_safety_level = 3
        self.max_dependencies = 2
        self.max_global_deadline = max_deadline

        
        self.state_dim = 3 + self.num_edges
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32)

        self.action_space = spaces.Dict({
            "location": spaces.Discrete(self.num_edges + 1), 
            "continuous_params": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        })

        self.server_loads = np.zeros(self.num_edges)
        self.task_queue = deque()
        self.current_task = None
        self.max_steps_per_episode = 200
        self.episode_step = 0

    def reset(self):
        self.episode_step = 0
        
        self.server_loads.fill(0)
        self.task_queue.clear()
        self._get_next_task()
        
        return self._get_state()


    def _generate_tasks(self, num_tasks):
        for _ in range(num_tasks):
            task_size = np.random.uniform(1.0, self.max_task_size)  #
            deadline = np.random.uniform(0.5, 2.0) * 1000  # ms
            safety_level = np.random.randint(1, self.max_safety_level + 1)
            dependencies = np.random.randint(0, self.max_dependencies + 1)
            
            task_complexity = np.random.uniform(500, 1500)
            deadline = np.random.uniform(0.5, 2.0) * 1000  # ms
            
            f_R = 1 - (deadline / self.max_global_deadline)
            f_S = (safety_level - 1) / (self.max_safety_level - 1) if self.max_safety_level > 1 else 0
            f_D = dependencies / self.max_dependencies if self.max_dependencies > 0 else 0
            priority_score = np.clip(self.w_realtime * f_R + self.w_safety * f_S + self.w_data * f_D, 0, 1)

            self.task_queue.append({
                "size": task_size,
                "complexity": task_complexity,  
                "deadline": deadline,
                "tau": priority_score,
                "battery": np.random.uniform(0.5, 1.0) 
            })
    def _get_next_task(self):
        """从队列中获取下一个任务"""
        if not self.task_queue:
            self._generate_tasks(self.num_devices * 2)  
        self.current_task = self.task_queue.popleft()
    def _get_state(self):
        
        channel_gains = np.random.uniform(0.5, 1.0, size=self.num_edges)
        norm_task_size = self.current_task["size"] / self.max_task_size
        norm_deadline = self.current_task["deadline"] / self.max_global_deadline
        norm_tau = self.current_task["tau"]
        norm_battery = self.current_task["battery"]  
        
        norm_server_loads = np.clip(self.server_loads / (self.C_edges * 10 + 1e-6), 0, 1)
        return np.concatenate([
            [norm_task_size, norm_deadline, norm_tau, norm_battery],
            norm_server_loads,
            channel_gains
        ]).astype(np.float32)

    def step(self, action):
        
        work_done = self.C_edges * self.time_step_duration * 100  
        self.server_loads = np.maximum(0, self.server_loads - work_done)

        
        location = action["location"]
        alpha = (action["continuous_params"][0] + 1) / 2
        beta = (action["continuous_params"][1] + 1) / 2

        d_n = self.current_task["size"] * 1024 * 1024 / 8
        c_n = self.current_task["complexity"]
        t_due = self.current_task["deadline"]
        tau_n = self.current_task["tau"]

        
        t_local = ((1 - alpha) * d_n * c_n) / (self.C_local * 1e9) 
        e_local = 1e-27 * ((1 - alpha) * d_n * c_n) * (self.C_local * 1e9) ** 2

        total_time = t_local
        total_energy = e_local

        if location > 0:  
            m_idx = location - 1
            t_comm = (alpha * d_n) / (self.B_up * 0.125)
            e_comm = 0.2 * t_comm  # p_max = 0.2W
            allocated_c = self.C_edges[m_idx] * max(beta, 0.01)
            t_mec = (alpha * d_n) / allocated_c
            e_mec = t_mec * 0.5 * beta  
            total_time = max(t_local, t_comm + t_mec)
            total_energy += (e_comm + e_mec)
            self.server_loads[m_idx] += (alpha * d_n)

        
        epsilon = 1e-6

        
        is_completed = 1 if total_time <= t_due else 0
        r_comp = is_completed * (self.R_base + self.w_tau * tau_n)

        delta_n = max(0, (total_time - t_due) / (t_due + epsilon))
        r_overdue = (self.P_base + self.w_tau * tau_n) * delta_n

        
        u_m = self.server_loads / (self.C_edges * 10 + epsilon)
        u_avg = np.mean(u_m)
        l_imb = np.mean(((u_m - u_avg) / (u_avg + epsilon)) ** 2)

        reward = r_comp - (self.w_O * r_overdue) - (self.w_E * total_energy) - (self.w_L * l_imb)

       
        self.episode_step += 1
        done = self.episode_step >= self.max_steps_per_episode
        self._get_next_task()
        next_state = self._get_state()

        return next_state, reward, done, {"is_success": is_completed}
