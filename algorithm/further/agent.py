import torch
import copy
import itertools
import torch.nn.functional as F
from algorithm.base import Base
from network.categorical_mlp import CategoricalMLP
from network.gaussian_mlp import GaussianMLP
from misc.rl_utils import to_transition, to_onehot, reparameterization
from misc.torch_utils import get_parameters
from torch.distributions import Categorical


class FURTHER(Base):
    """Class for FURTHER agent
    Args:
        env (gym env): Gym environment from gym_env folder
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
    """
    def __init__(self, env, log, tb_writer, args, name, i_agent) -> None:
        super(FURTHER, self).__init__(env, log, tb_writer, args, name, i_agent)

        self._set_actor()
        self._set_critic()
        self._set_gain()
        self._set_inference()
        self._set_optimizer()

    def _set_actor(self) -> None:
        self.actor = CategoricalMLP(
            n_input=self.env_n_obs + self.args.n_latent,
            n_output=self.env_n_action,
            name=self.name + "_actor",
            log=self.log, args=self.args, device=self.device)

    def _set_critic(self) -> None:
        self.critic1 = CategoricalMLP(
            n_input=self.env_n_obs + self.args.n_latent + (self.args.n_agent - 1) * self.env_n_action,
            n_output=self.env_n_action,
            name=self.name + "_critic1",
            log=self.log, args=self.args, device=self.device)
        self.critic_target1 = copy.deepcopy(self.critic1)

        self.critic2 = CategoricalMLP(
            n_input=self.env_n_obs + self.args.n_latent + (self.args.n_agent - 1) * self.env_n_action,
            n_output=self.env_n_action,
            name=self.name + "_critic2",
            log=self.log, args=self.args, device=self.device)
        self.critic_target2 = copy.deepcopy(self.critic2)

    def _set_gain(self) -> None:
        gain = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.gain = torch.nn.Parameter(gain, requires_grad=True)

    def _set_inference(self) -> None:
        self.encoder = GaussianMLP(
            n_input=self.env_n_obs * 2 + self.args.n_agent * self.env_n_action + 1 + self.args.n_latent,
            n_output=self.args.n_latent,
            name=self.name + "_encoder",
            log=self.log, args=self.args, device=self.device)

        self.decoder = CategoricalMLP(
            n_input=self.env_n_obs + self.args.n_latent,
            n_output=self.env_n_action,
            name=self.name + "_decoder",
            log=self.log, args=self.args, device=self.device)

    def _set_optimizer(self) -> None:
        # Set actor optimizer
        self.actor_optimizer = torch.optim.Adam(get_parameters(self.actor), lr=self.args.actor_lr)

        # Set critic optimizer
        critic_params = itertools.chain(
            get_parameters(self.critic1),
            get_parameters(self.critic2))
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.args.critic_lr)

        # Set gain optimizer
        self.gain_optimizer = torch.optim.Adam(get_parameters(self.gain), lr=self.args.gain_lr)

        # Set inference optimizer
        inference_params = itertools.chain(
            get_parameters(self.encoder),
            get_parameters(self.decoder))
        self.inference_optimizer = torch.optim.Adam(inference_params, lr=self.args.inference_lr)

    def encode(self, peer_latent, obs, actions, reward, next_obs):
        transition = to_transition(obs, actions, reward, next_obs, self, self.args)
        encoder_input = torch.cat([peer_latent, transition], dim=-1)
        encoder_mu, encoder_logvar = self.encoder(encoder_input)
        next_peer_latent = reparameterization(encoder_mu, encoder_logvar)

        return next_peer_latent

    def _decode(self, obs, peer_latent):
        decoder_input = torch.cat([obs, peer_latent], dim=-1)
        decoder_out = self.decoder(decoder_input)
        peer_action_prob = F.softmax(decoder_out, dim=-1)
        return peer_action_prob

    def act(self, obs, peer_latent, timestep):
        # Compute output of policy
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actor_input = torch.cat([obs, peer_latent], dim=-1)
        logit = self.actor(actor_input)

        # Select action
        action = self._select_action(logit, action_selection="softmax")

        return action

    def add_transition(self, obs, peer_latents, actions, rewards, next_obs, next_peer_latents):
        self.memory.push(
            obs=obs,
            peer_latent=peer_latents[self.i_agent],
            actions=actions,
            reward=rewards[self.i_agent],
            next_obs=next_obs,
            next_peer_latent=next_peer_latents[self.i_agent])

    def _get_actor_loss(self, timestep):
        if timestep < self.args.batch_size:
            self.loss["actor"] = 0.
            return

        # Process transition
        obs, peer_latent, actions, _, _, _ = self.memory.sample(mode=self.args.sampling_mode)

        # Get Q-value
        peer_action_onehot = to_onehot(actions[..., 1 - self.i_agent], dim=self.env_n_action).float()
        critic_input = torch.cat([obs, peer_latent, peer_action_onehot], dim=-1)
        q_value1 = self.critic1(critic_input)
        q_value2 = self.critic2(critic_input)
        q_value = torch.min(q_value1, q_value2)

        # Get log_prob
        actor_input = torch.cat([obs, peer_latent], dim=-1).detach()
        logit = self.actor(actor_input)
        action_prob = F.softmax(logit, dim=-1)
        action_logprob = F.log_softmax(logit, dim=-1)

        # Get actor loss
        assert action_prob.shape == q_value.shape, "{} vs {}".format(action_prob.shape, q_value.shape)
        assert action_logprob.shape == q_value.shape, "{} vs {}".format(action_logprob.shape, q_value.shape)
        entropy = self.args.entropy_weight * (-action_prob * action_logprob)
        self.loss["actor"] = -(action_prob * q_value.detach() + entropy).sum(dim=-1).mean()

        # For logging
        self.tb_writer.add_scalars("loss/entropy", {"agent" + str(self.i_agent): entropy.sum(dim=-1).mean()}, timestep)

    def _get_critic_loss(self, timestep):
        if timestep < self.args.batch_size:
            _, _, _, reward, _, _ = self.memory.sample(mode="all")
            if torch.max(reward) > self.gain:
                self.gain.data = torch.max(reward)
                self.log[self.args.log_name].info("[{}] Setting initial gain: {}".format(self.name, self.gain))
            self.loss["critic"] = 0.
            return

        # Process transition
        obs, peer_latent, actions, reward, next_obs, _ = self.memory.sample(mode=self.args.sampling_mode)

        # Get next_peer_latent from current encoder
        actions_onehot = [to_onehot(actions[:, i_agent], dim=self.env_n_action) for i_agent in range(self.args.n_agent)]
        actions_onehot = torch.cat(actions_onehot, dim=-1).float()
        encoder_input = torch.cat([peer_latent, obs, actions_onehot, reward, next_obs], dim=-1)
        encoder_mu, encoder_logvar = self.encoder(encoder_input)
        next_peer_latent = reparameterization(encoder_mu, encoder_logvar).detach()

        # Get Q-value
        peer_action_onehot = to_onehot(actions[..., 1 - self.i_agent], dim=self.env_n_action).float()
        critic_input = torch.cat([obs, peer_latent, peer_action_onehot], dim=-1).detach()
        q_value1 = self.critic1(critic_input).gather(index=actions[..., self.i_agent].unsqueeze(1), dim=-1)
        q_value2 = self.critic2(critic_input).gather(index=actions[..., self.i_agent].unsqueeze(1), dim=-1)

        # Get own agent's next action probability
        next_actor_input = torch.cat([next_obs, next_peer_latent], dim=-1)
        next_logit = self.actor(next_actor_input)
        next_action_prob = F.softmax(next_logit, dim=-1)
        next_action_logprob = F.log_softmax(next_logit, dim=-1)

        # Get next_peer_action
        next_peer_action_prob = self._decode(next_obs, next_peer_latent)

        # Get next Q-value
        next_q_value = 0.
        for next_peer_action in range(self.env_n_action):
            next_peer_action = torch.tensor([next_peer_action], dtype=torch.int64, device=self.device)
            next_peer_action_onehot = to_onehot(next_peer_action, dim=self.env_n_action).float().repeat(obs.shape[0], 1)
            next_critic_input = torch.cat([next_obs, next_peer_latent, next_peer_action_onehot], dim=-1).detach()
            next_q_value1 = self.critic_target1(next_critic_input)
            next_q_value2 = self.critic_target2(next_critic_input)
            next_peer_action_prob_repeat = next_peer_action_prob[..., next_peer_action].repeat(1, self.env_n_action)
            next_q_value += next_peer_action_prob_repeat * torch.minimum(next_q_value1, next_q_value2)

        # Get critic loss
        next_entropy = self.args.entropy_weight * (-next_action_prob * next_action_logprob)
        next_value = (next_action_prob * next_q_value + next_entropy).sum(dim=-1).reshape(-1, 1)
        target = reward - self.gain + next_value.detach()
        assert q_value1.shape == target.shape, "{} vs {}".format(q_value1.shape, target.shape)
        self.loss["critic"] = torch.square(q_value1 - target).mean() + torch.square(q_value2 - target).mean()

        # For logging
        self.tb_writer.add_scalars("debug/gain", {"agent" + str(self.i_agent): self.gain}, timestep)

    def _get_inference_loss(self, timestep):
        if timestep < self.args.batch_size:
            self.loss["inference"], self.loss["kl"] = 0., 0.
            return

        # Set initial prior flag
        is_initial_prior = True if timestep == self.args.batch_size else False

        # Perform encoding
        obs, peer_latent, actions, reward, next_obs, _ = self.memory.sample(mode="all")
        actions_onehot = [to_onehot(actions[:, i_agent], dim=self.env_n_action) for i_agent in range(self.args.n_agent)]
        actions_onehot = torch.cat(actions_onehot, dim=-1).float()
        encoder_input = torch.cat([peer_latent, obs, actions_onehot, reward, next_obs], dim=-1)[:-1, :]
        encoder_mu, encoder_logvar = self.encoder(encoder_input)
        next_peer_latent = reparameterization(encoder_mu, encoder_logvar)

        # Perform decoding
        decoder_input = torch.cat([obs[1:, :], next_peer_latent], dim=-1)
        decoder_out = self.decoder(decoder_input)
        next_peer_prob = F.softmax(decoder_out, dim=-1)
        next_peer_dist = Categorical(probs=next_peer_prob)

        # Get reconstruction loss
        next_peer_action = actions[1:, 1 - self.i_agent]
        assert next_peer_dist.sample().shape == next_peer_action.shape
        next_peer_logprob = next_peer_dist.log_prob(next_peer_action)
        self.loss["inference"] = -torch.mean(next_peer_logprob)

        # Get KLD error
        # KL(N(mu,E), N(m, S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m - mu)^T S^-1 (m - mu)))
        # Ref: https://github.com/lmzintgraf/varibad/blob/master/vae.py
        if is_initial_prior:
            encoder_mu = torch.cat([torch.zeros((1, self.args.n_latent), device=self.device), encoder_mu], dim=0)
            encoder_logvar = torch.cat([torch.zeros((1, self.args.n_latent), device=self.device), encoder_logvar], dim=0)

        kl_first_term = torch.sum(encoder_logvar[:-1, :], dim=-1) - torch.sum(encoder_logvar[1:, :], dim=-1)
        kl_second_term = self.args.n_latent
        kl_third_term = torch.sum(1. / torch.exp(encoder_logvar[:-1, :]) * torch.exp(encoder_logvar[1:, :]), dim=-1)
        kl_fourth_term = \
            (encoder_mu[:-1, :] - encoder_mu[1:, :]) / torch.exp(encoder_logvar[:-1, :]) * \
            (encoder_mu[:-1, :] - encoder_mu[1:, :])
        kl_fourth_term = kl_fourth_term.sum(dim=-1)
        kl = 0.5 * (kl_first_term - kl_second_term + kl_third_term + kl_fourth_term)
        self.loss["kl"] = self.args.kl_weight * torch.mean(kl)

    def get_loss(self, agents, timestep):
        # Initialize loss
        self.loss = {}

        # Get actor and inference loss
        self._get_actor_loss(timestep)
        self._get_critic_loss(timestep)
        self._get_inference_loss(timestep)

        # For logging
        for key, value in self.loss.items():
            self.tb_writer.add_scalars("loss/" + key, {"agent" + str(self.i_agent): value}, timestep)

        return self.loss

    def update(self, loss):
        if isinstance(loss["actor"], torch.Tensor):
            self.actor_optimizer.zero_grad()
            loss["actor"].backward()
            torch.nn.utils.clip_grad_norm_(get_parameters(self.actor), self.args.max_grad_clip)
            self.actor_optimizer.step()

        if isinstance(loss["critic"], torch.Tensor):
            self.critic_optimizer.zero_grad()
            self.gain_optimizer.zero_grad()
            loss["critic"].backward()
            critic_params = itertools.chain(
                get_parameters(self.critic1),
                get_parameters(self.critic2))
            torch.nn.utils.clip_grad_norm_(critic_params, self.args.max_grad_clip)
            torch.nn.utils.clip_grad_norm_(get_parameters(self.gain), self.args.max_grad_clip)
            self.critic_optimizer.step()
            self.gain_optimizer.step()

        if isinstance(loss["inference"], torch.Tensor) or isinstance(loss["kl"], torch.Tensor):
            self.inference_optimizer.zero_grad()
            (loss["inference"] + loss["kl"]).backward()
            inference_params = itertools.chain(
                get_parameters(self.encoder),
                get_parameters(self.decoder))
            torch.nn.utils.clip_grad_norm_(inference_params, self.args.max_grad_clip)
            self.inference_optimizer.step()

        with torch.no_grad():
            for p, p_target in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                p_target.data.mul_(self.args.polyak)
                p_target.data.add_((1. - self.args.polyak) * p.data)

            for p, p_target in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                p_target.data.mul_(self.args.polyak)
                p_target.data.add_((1. - self.args.polyak) * p.data)
