import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from gat import GAT

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action, hidden_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.max_action = max_action

    def forward(self, obs):
        action = self.l1(obs)
        action = self.max_action * torch.tanh(action)

        return action


class LocalCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        # Individual Q1 architecture
        self.q1_fc_o_i = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Individual Q2 architecture
        self.q2_fc_o_i = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_individual_q(self, obs, a):
        oa = torch.cat([obs, a], -1)
        q1 = self.q1_fc_o_i(oa)
        q2 = self.q2_fc_o_i(oa)
        return q1, q2

    def Q1_individual(self, obs, a):
        oa = torch.cat([obs, a], -1)
        q1 = self.q1_fc_o_i(oa)
        return q1


class GlobalCritic(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_dim, vae_dim):
        super(GlobalCritic, self).__init__()
        # Global Q1 architecture
        self.q1_fc_o = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.q1_fc_s = nn.Sequential(
            nn.Linear((state_dim + action_dim), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        self.q1_gat = GAT(n_feat=hidden_dim // 2, n_class=hidden_dim // 2, n_heads=2, n_hid=hidden_dim // 2)
        self.q1_v = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Global Q2 architecture
        self.q2_fc_o = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.q2_fc_s = nn.Sequential(
            nn.Linear((state_dim + action_dim), hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        self.q2_gat = GAT(n_feat=hidden_dim // 2, n_class=hidden_dim // 2, n_heads=2, n_hid=hidden_dim // 2)
        self.q2_v = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Baseline architecture
        self.fc_o = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_s = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        self.v_gat = GAT(n_feat=hidden_dim // 2, n_class=hidden_dim // 2, n_heads=2, n_hid=hidden_dim // 2)
        self.fc_v = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # VAE architecture
        self.vae_input_dim = obs_dim + state_dim*2 + action_dim*2
        self.vae_fc_vae = nn.Sequential(
            nn.Linear(self.vae_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.vae_fc_mean = nn.Linear(hidden_dim, vae_dim)
        self.vae_fc_logvar = nn.Linear(hidden_dim, vae_dim)
        self.vae_fc_recons = nn.Sequential(
            nn.Linear(vae_dim+state_dim*2+action_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim+1)
        )

    def forward_global_q(self, obs, a, global_s, global_a, global_mask):
        oa = torch.cat([obs, a], -1)
        sa = torch.cat([global_s, global_a], -1)

        q1 = self.q1_fc_o(oa)
        q1_ = self.q1_fc_s(sa)
        q1_ = self.q1_gat(q1_, global_mask)
        q1_ = q1_[:, 0, :]
        q1 = self.q1_v(torch.cat([q1, q1_], -1))

        q2 = self.q2_fc_o(oa)
        q2_ = self.q2_fc_s(sa)
        q2_ = self.q2_gat(q2_, global_mask)
        q2_ = q2_[:, 0, :]
        q2 = self.q2_v(torch.cat([q2, q2_], -1))
        return q1, q2

    def Q1_global(self, obs, a, global_s, global_a, global_mask):
        oa = torch.cat([obs, a], -1)
        sa = torch.cat([global_s, global_a], -1)

        q1 = self.q1_fc_o(oa)
        q1_ = self.q1_fc_s(sa)
        q1_ = self.q1_gat(q1_, global_mask)
        q1_ = q1_[:, 0, :]
        q1 = self.q1_v(torch.cat([q1, q1_], -1))
        return q1

    def V(self, obs, global_s, global_a, global_mask):
        baseline_a = global_a.clone()
        baseline_a[:, 0, :] = (torch.zeros_like(baseline_a[:, 0, :]) - 1).to(global_a.device)
        sa = torch.cat([global_s, baseline_a], -1)

        v = self.fc_o(obs)
        v_ = self.fc_s(sa)
        v_ = self.v_gat(v_, global_mask)
        v_ = v_[:, 0, :]
        v = self.fc_v(torch.cat([v, v_], -1))
        return v

    def forward_vae_encode(self, obs, s, a, global_s, global_a):
        obs = obs.unsqueeze(1).repeat(1, global_s.size(1), 1)
        s = s.unsqueeze(1).repeat(1, global_s.size(1), 1)
        a = a.unsqueeze(1).repeat(1, global_s.size(1), 1)
        x = torch.cat([obs, s, a, global_s, global_a], -1)
        x = self.vae_fc_vae(x)
        mu = self.vae_fc_mean(x)
        logvar = self.vae_fc_logvar(x)

        return mu, logvar

    def forward_vae(self, obs, s, a, global_s, global_a):
        mu, logvar = self.forward_vae_encode(obs, s, a, global_s, global_a)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        s = s.unsqueeze(1).repeat(1, global_s.size(1), 1)
        a = a.unsqueeze(1).repeat(1, global_s.size(1), 1)
        x = torch.cat([s, a, global_s, global_a], -1)
        recons = self.vae_fc_recons(torch.cat([z, x], -1))
        return recons, mu, logvar


class AblationV2(object):
    """
    Ablation Model V2 that removes the VAEs from the centralized critic
    """
    def __init__(
            self,
            obs_dim,
            state_dim,
            action_dim,
            max_action,
            hidden_dim,
            max_agents,
            vae_dim=20,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=3e-4,
            critic_lr=5e-4,
            weight_fac=1,
            device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor = Actor(obs_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.local_critic = LocalCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.local_critic_target = copy.deepcopy(self.local_critic)
        self.local_critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), lr=critic_lr*1.5)

        self.global_critic = GlobalCritic(obs_dim, state_dim, action_dim, hidden_dim, vae_dim).to(self.device)
        self.global_critic_target = copy.deepcopy(self.global_critic)
        self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.weight_fac = weight_fac

        self.total_it = 0

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        # Sample replay buffer
        (obs, action, next_obs, reward, global_reward, not_done, global_s, global_a, global_mask, next_global_s, next_global_a, next_global_mask) = replay_buffer.sample(batch_size)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            next_global_a[:, 0, :] = next_action
            # Compute the target Q value
            target_Q1_i, target_Q2_i = self.local_critic_target.forward_individual_q(next_obs, next_action)
            target_Q1_g, target_Q2_g = self.global_critic_target.forward_global_q(next_obs, next_action, next_global_s, next_global_a, next_global_mask)
            target_Q_i = torch.min(target_Q1_i, target_Q2_i)
            target_Q_g = torch.min(target_Q1_g, target_Q2_g)
            target_Q_i = reward + not_done * self.discount * target_Q_i
            target_Q_g = global_reward + not_done * self.discount * target_Q_g
            # Compute the target V value
            target_V = self.global_critic_target.V(next_obs, next_global_s, next_global_a, global_mask)
            target_V = global_reward + not_done * self.discount * target_V

        # Get current Q estimates
        current_Q1_i, current_Q2_i = self.local_critic.forward_individual_q(obs, action)
        current_Q1_g, current_Q2_g = self.global_critic.forward_global_q(obs, action, global_s, global_a, global_mask)
        # Get current V estimates
        current_V = self.global_critic.V(obs, global_s, global_a, global_mask)

        # Train VAE
        recon_obs, mu, logvar = self.global_critic.forward_vae(obs, global_s[:, 0, :], action, global_s, global_a)
        mse = F.mse_loss(recon_obs, torch.cat([next_obs, reward], -1).unsqueeze(1).repeat(1, recon_obs.size(1), 1))
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = mse + kld

        # Compute critic loss
        local_critic_loss = F.mse_loss(current_Q1_i, target_Q_i) + F.mse_loss(current_Q2_i, target_Q_i)
        global_critic_loss = (F.mse_loss(current_Q1_g, target_Q_g) + F.mse_loss(current_Q2_g, target_Q_g)
                              + F.mse_loss(current_V, target_V) + 0.01 * vae_loss)

        # Optimize the critic
        self.local_critic_optimizer.zero_grad()
        self.global_critic_optimizer.zero_grad()
        local_critic_loss.backward()
        global_critic_loss.backward()
        self.local_critic_optimizer.step()
        self.global_critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss_i = - self.local_critic.Q1_individual(obs, self.actor(obs))
            actor_loss_g = - (self.global_critic.Q1_global(obs, self.actor(obs), global_s, global_a, global_mask) - current_V.detach())
            actor_loss = (actor_loss_i + self.weight_fac * actor_loss_g).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.local_critic.parameters(), self.local_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.global_critic.parameters(), self.global_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (torch.min(current_Q1_i, current_Q2_i).mean().detach().cpu().numpy(),
                torch.min(current_Q1_g, current_Q2_g).mean().detach().cpu().numpy(),
                current_V.mean().detach().cpu().numpy(),
                F.mse_loss(current_Q1_i, target_Q_i).detach().cpu().numpy(),
                F.mse_loss(current_Q1_g, target_Q_g).detach().cpu().numpy(),
                F.mse_loss(current_V, target_V).detach().cpu().numpy(),
                vae_loss.detach().cpu().numpy())

    def save(self, filename):
        torch.save(self.local_critic.state_dict(), filename + "_local_critic")
        torch.save(self.local_critic_optimizer.state_dict(), filename + "_local_critic_optimizer")

        torch.save(self.global_critic.state_dict(), filename + "_global_critic")
        torch.save(self.global_critic_optimizer.state_dict(), filename + "_global_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def save_best(self, filename):
        torch.save(self.local_critic.state_dict(), filename + "_local_critic")
        torch.save(self.global_critic.state_dict(), filename + "_global_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.local_critic.load_state_dict(torch.load(filename + "_local_critic"))
        self.local_critic_optimizer.load_state_dict(torch.load(filename + "_local_critic_optimizer"))
        self.local_critic_target = copy.deepcopy(self.local_critic)

        self.global_critic.load_state_dict(torch.load(filename + "_global_critic"))
        self.global_critic_optimizer.load_state_dict(torch.load(filename + "_global_critic_optimizer"))
        self.global_critic_target = copy.deepcopy(self.global_critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def load_best(self, filename):
        success_high = 0
        best_file_name = None
        for file_name in os.listdir(filename):
            if "best_model" in file_name:
                if "actor" in file_name:
                    val = float(file_name.split("_")[6])
                    if val >= success_high:
                        success_high = val
                        best_file_name = os.path.join(filename, file_name.replace("_actor", ""))

        print("Load best model: {}".format(best_file_name))
        self.local_critic.load_state_dict(torch.load(best_file_name + "_local_critic"))
        self.local_critic_target = copy.deepcopy(self.local_critic)

        self.global_critic.load_state_dict(torch.load(best_file_name + "_global_critic"))
        self.global_critic_target = copy.deepcopy(self.global_critic)

        self.actor.load_state_dict(torch.load(best_file_name + "_actor"))
        self.actor_target = copy.deepcopy(self.actor)


if __name__ == '__main__':
    o_ = torch.randn((10, 91))
    a_ = torch.randn((10, 2))
    g_s = torch.randn((10, 30, 10))
    g_a = torch.randn((10, 30, 2))
    g_mask = torch.randn(10, 30)

    model = AblationV2(obs_dim=91, action_dim=2, state_dim=10, hidden_dim=128, max_action=1)

    qvalue1, qvalue2 = model.local_critic.forward_individual_q(o_, a_)
    print(qvalue1.shape, qvalue2.shape)
    qvalue1, qvalue2 = model.global_critic.forward_global_q(o_, a_, g_s, g_a, g_mask)
    print(qvalue1.shape, qvalue2.shape)
    qvalue1 = model.local_critic.Q1_individual(o_, a_)
    print(qvalue1.shape)
    qvalue1 = model.global_critic.Q1_global(o_, a_, g_s, g_a, g_mask)
    print(qvalue1.shape)
    value = model.global_critic.V(o_, g_s, g_a, g_mask)
    print(value.shape)
