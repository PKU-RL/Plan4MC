from __future__ import annotations

import torch
import torch.nn as nn

from mineclip.utils import build_mlp
from .distribution import MultiCategorical


class MultiCategoricalActor(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action_dim: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.preprocess = preprocess_net
        for action in action_dim:
            net = build_mlp(
                input_dim=preprocess_net.output_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
            )
            self.mlps.append(net)
        self._action_dim = action_dim
        self._device = device

    def forward(self, x, state=None, info=None):
        hidden = None
        y, _ = self.preprocess(x)
        return torch.cat([mlp(y) for mlp in self.mlps], dim=1), hidden

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)


"""
# my implementation, actor and critic share the same feature encoder
class MultiCategoricalActorCritic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action_dim: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.preprocess = preprocess_net
        for action in action_dim:
            net = build_mlp(
                input_dim=preprocess_net.output_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
            )
            self.mlps.append(net)
        self._action_dim = action_dim
        self._device = device

        # value function
        self.v = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None,
        )
        #print(self.mlps)

    # actor forward
    def forward(self, x, state=None, info=None):
        hidden = None
        x, _ = self.preprocess(x)
        '''
        for mlp in self.mlps:
            #print(mlp)
            print(x.shape, x.min(), x.max())
            a = mlp(x)
            print(a)
        '''
        return torch.cat([mlp(x) for mlp in self.mlps], dim=1), hidden

    # critic forward
    def forward_critic(self, x):
        x, _ = self.preprocess(x)
        return torch.squeeze(self.v(x), -1)

    # actor critic forward
    def forward_actor_critic(self, x):
        x, _ = self.preprocess(x)
        act_logits = torch.cat([mlp(x) for mlp in self.mlps], dim=1)
        val = torch.squeeze(self.v(x), -1)
        return act_logits, val

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)
"""

# my implementation 9-7, actor and critic don't share parameters

class Critic(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action_dim: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        #self.mlps = nn.ModuleList()
        self.preprocess = preprocess_net
        '''
        for action in action_dim:
            net = build_mlp(
                input_dim=preprocess_net.output_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
            )
            self.mlps.append(net)
        self._action_dim = action_dim
        '''
        self._device = device

        self.v = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None,
        )

    # critic forward
    def forward(self, x):
        y, _ = self.preprocess(x)
        return torch.squeeze(self.v(y), -1)
