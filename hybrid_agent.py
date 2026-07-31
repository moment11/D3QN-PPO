from pathlib import Path

from D3QN import D3QN
from PPO import PPO


class D3QNPPOAgent:
    """D3QN macro policy with one destination-specific PPO per action."""

    def __init__(self, state_dim, num_locations, checkpoint_dir="checkpoints",
                 d3qn_lr=3e-4, ppo_actor_lr=3e-4, ppo_critic_lr=1e-3,
                 gamma=0.99, ppo_epochs=10, ppo_clip=0.2,
                 batch_size=256, target_update_interval=1000,
                 tper_alpha=0.6, tper_beta=0.4, tper_lambda=0.001,
                 seed=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.d3qn_meta = D3QN(
            alpha=d3qn_lr,
            state_dim=state_dim,
            action_dim=num_locations,
            fc1_dim=256,
            fc2_dim=256,
            ckpt_dir=str(self.checkpoint_dir),
            gamma=gamma,
            batch_size=batch_size,
            target_update_interval=target_update_interval,
            tper_alpha=tper_alpha,
            tper_beta=tper_beta,
            tper_lam=tper_lambda,
            seed=seed,
        )
        self.ppo_sub_policies = {
            location: PPO(
                state_dim=state_dim,
                action_dim=2,
                lr_actor=ppo_actor_lr,
                lr_critic=ppo_critic_lr,
                gamma=gamma,
                k_epochs=ppo_epochs,
                eps_clip=ppo_clip,
                has_continuous_action_space=True,
            )
            for location in range(num_locations)
        }

    def select_action(self, state, is_train=True):
        location = self.d3qn_meta.choose_action(state, is_train=is_train)
        continuous_action = self.ppo_sub_policies[location].select_action(
            state, record=is_train, deterministic=not is_train
        )
        return location, continuous_action

    def store_transition(self, state, location, reward, next_state, done):
        self.d3qn_meta.remember(state, location, reward, next_state, done)
        policy = self.ppo_sub_policies[location]
        policy.buffer.rewards.append(reward)
        policy.buffer.is_terminals.append(done)

    def learn(self, ppo_update_timestep):
        statistics = self.d3qn_meta.learn() or {}
        for location, policy in self.ppo_sub_policies.items():
            if len(policy.buffer.states) >= ppo_update_timestep:
                ppo_stats = policy.update()
                if ppo_stats:
                    statistics[f"ppo_{location}_loss"] = ppo_stats["ppo_loss"]
        return statistics

    def flush_ppo(self):
        statistics = {}
        for location, policy in self.ppo_sub_policies.items():
            if policy.buffer.states:
                ppo_stats = policy.update()
                if ppo_stats:
                    statistics[f"ppo_{location}_loss"] = ppo_stats["ppo_loss"]
        return statistics

    def save(self, label):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.d3qn_meta.save_models(label)
        for location, policy in self.ppo_sub_policies.items():
            policy.save(self.checkpoint_dir / f"ppo_location_{location}_{label}.pth")


DHAL_Agent = D3QNPPOAgent