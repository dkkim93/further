import torch
import torch.nn.functional as F
import numpy as np
from algorithm.base import Base
from misc.torch_utils import get_parameters


class TabularQ(Base):
    """Class for tabular Q agent
    Args:
        env (gym env): Gym environment from gym_env folder
        log (dict): Dictionary that contains python logging
        tb_writer (SummeryWriter): Used for tensorboard logging
        args (argparse): Python argparse that contains arguments
        name (str): Specifies agent's name
        i_agent (int): Agent index among the agents in the shared environment
    """
    def __init__(self, env, log, tb_writer, args, name, i_agent) -> None:
        super(TabularQ, self).__init__(env, log, tb_writer, args, name, i_agent)

        self._set_actor()
        self.epsilon = 0.05
        self.discount = 0.9

    def _set_actor(self) -> None:
        persona = np.array([10.0, 20.0])
        self.actor = torch.nn.Parameter(torch.from_numpy(persona).float().to(self.device), requires_grad=True)
        self.actor_optimizer = torch.optim.SGD(get_parameters(self.actor), lr=0.5)

    def act(self, obs, peer_latent, timestep):
        # Compute Q-values
        logit = self.actor

        # Select action
        action = self._select_action(logit, action_selection="epsilon")

        return action

    def encode(self, *args, **kwargs):
        return torch.zeros((1, self.args.n_latent), dtype=torch.float32, device=self.device)

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
        _, _, actions, reward, _, _ = self.memory.sample(mode="recent")

        # Get Q-value
        q_value = self.actor[..., actions[..., self.i_agent]].reshape(-1, 1)

        # Get next Q-value
        next_q_value = torch.max(self.actor).reshape(-1, 1)

        # Get actor loss
        target = reward + self.discount * next_q_value
        assert q_value.shape == target.shape, "{} vs {}".format(q_value.shape, target.shape)
        self.loss["actor"] = F.mse_loss(q_value, target.detach())

    def get_loss(self, agent, timestep):
        # Initialize loss
        self.loss = {}

        # Get actor loss
        self._get_actor_loss(timestep)

        # For logging
        for key, value in self.loss.items():
            self.tb_writer.add_scalars("loss/" + key, {"agent" + str(self.i_agent): value}, timestep)

        return self.loss

    def update(self, loss):
        loss = sum(loss.values())
        if not isinstance(loss, torch.Tensor):
            return

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(get_parameters(self.actor), self.args.max_grad_clip)
        self.actor_optimizer.step()
