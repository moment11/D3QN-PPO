import torch
import numpy as np
from collections import deque, namedtuple

from D3QN import D3QN
from PPO import PPO


class DHAL_Agent:
    def __init__(self,
                 d3qn_state_dim,
                 d3qn_action_dim,
                 d3qn_ckpt_dir,
                 ppo_state_dim,
                 ppo_action_dim,
                 ppo_lr_actor,
                 ppo_lr_critic,
                 ppo_gamma,
                 ppo_k_epochs,
                 ppo_eps_clip,
                 
                 num_locations):
        
        self.d3qn_meta = D3QN(alpha=0.0003,state_dim=d3qn_state_dim,action_dim=d3qn_action_dim,fc1_dim=256,fc2_dim=256,ckpt_dir=d3qn_ckpt_dir)
        self.ppo_sub_policies = {}
        for i in range(num_locations):
            
            self.ppo_sub_policies[i] = PPO(state_dim=ppo_state_dim,
                                           action_dim=ppo_action_dim,
                                           lr_actor=ppo_lr_actor,
                                           lr_critic=ppo_lr_critic,
                                           gamma=ppo_gamma,
                                           K_epochs=ppo_k_epochs,
                                           eps_clip=ppo_eps_clip,
                                           has_continuous_action_space=True)

    def select_action(self, state, isTrain=True):
        """
        分层决策流程
        """
        
        discrete_action = self.d3qn_meta.choose_action(state, isTrain)

        
        
        ppo_policy = self.ppo_sub_policies[discrete_action]
        continuous_action = ppo_policy.select_action(state)  

        return discrete_action, continuous_action

    def store_transition(self, state, discrete_action, reward, next_state, done):
        
        self.d3qn_meta.remember(state, discrete_action, reward, next_state, done)

        
        ppo_policy = self.ppo_sub_policies[discrete_action]
        ppo_policy.buffer.rewards.append(reward)
        ppo_policy.buffer.is_terminals.append(done)

    def learn(self, ppo_update_timestep, d3qn_target_update_interval=1000):
       
        
        self.d3qn_meta.learn()

        for i, ppo_policy in self.ppo_sub_policies.items():
            if len(ppo_policy.buffer.states) >= ppo_update_timestep:
                print(f"--- Updating PPO sub-policy for location {i} ---")
                ppo_policy.update()
                ppo_policy.decay_action_std(action_std_decay_rate=0.01, min_action_std=0.1)
