import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
from torch.optim.lr_scheduler import StepLR

if torch.cuda.is_available():
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Sigmoid()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, deterministic=False):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = action_mean if deterministic and self.has_continuous_action_space else dist.sample()
        if self.has_continuous_action_space:
            action = torch.clamp(action, 0.0, 1.0)
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ], lr=lr_actor, weight_decay=0.0)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, record=True, deterministic=False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(
                    state, deterministic=deterministic
                )

            if record:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        if not self.buffer.rewards:
            return None
        # Monte Carlo estimate of returns
        advantages = []
        gae = 0.0
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            is_terminal = self.buffer.is_terminals[i]
            state_value = float(self.buffer.state_values[i].item())
            next_state_value = 0.0
            if i < len(self.buffer.rewards) - 1:
                next_state_value = float(self.buffer.state_values[i + 1].item())

            if is_terminal:
                td_error = reward - state_value
                gae = td_error
            else:
                # TD error = r_t + gamma * V(s_{t+1}) - V(s_t)
                td_error = reward + self.gamma * next_state_value - state_value
                # GAE(gamma, lambda) = td_error + gamma * lambda * GAE_{t+1}
                gae = td_error + self.gamma * 0.95 * gae

            advantages.insert(0, gae)

        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        old_state_values_tensor = torch.stack(
            self.buffer.state_values, dim=0
        ).reshape(-1).detach().to(device)
        returns = advantages + old_state_values_tensor

        
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-7
        )

        

        
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(
            self.buffer.logprobs, dim=0
        ).reshape(-1).detach().to(device)

       
        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.reshape(-1)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        
        self.scheduler.step()
        return {"ppo_loss": float(loss.mean().item())}
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        #
        # # Normalizing the rewards
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        #
        # # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        # old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        # old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        #
        # # calculate advantages
        # advantages = rewards.detach() - old_state_values.detach()
        #
        # # Optimize policy for K epochs
        # for _ in range(self.K_epochs):
        #
        #     # Evaluating old actions and values
        #     logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        #
        #     # match state_values tensor dimensions with rewards tensor
        #     state_values = torch.squeeze(state_values)
        #
        #     # Finding the ratio (pi_theta / pi_theta__old)
        #     ratios = torch.exp(logprobs - old_logprobs.detach())
        #
        #     # Finding Surrogate Loss
        #     surr1 = ratios * advantages
        #     surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        #
        #     # final loss of clipped objective PPO
        #     loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
        #
        #     # take gradient step
        #     self.optimizer.zero_grad()
        #     loss.mean().backward()
        #     self.optimizer.step()
        #
        # # Copy new weights into old policy
        # self.policy_old.load_state_dict(self.policy.state_dict())
        #
        # # clear buffer
        # self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)
        
        
       


