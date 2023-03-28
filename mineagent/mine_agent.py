from __future__ import annotations

import torch.nn as nn

from .batch import Batch


# my implementation 9-7, actor and critic don't share parameters
class MineAgent(nn.Module):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        deterministic_eval: bool = False, # use stochastic in both exploration and test
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self._deterministic_eval = deterministic_eval
        self.dist_fn = actor.dist_fn

    # forward actor
    def forward(
        self,
        batch: Batch,
        state=None,
        **kwargs,
    ) -> Batch:
        logits, hidden = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = dist.mode()
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    '''
    # input an obs, output the action distribution
    def _distribution(self, obs):
        logits, _ = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        return dist
    '''


    # forward actor critic
    def forward_actor_critic(
        self,
        batch: Batch
    ) -> Batch:
        logits, _ = self.actor(batch.obs)
        val = self.critic(batch.obs)
        
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = dist.mode()
        else:
            act = dist.sample()
        logp = dist.log_prob(act)

        return Batch(logits=logits, act=act, dist=dist, logp=logp, val=val)

