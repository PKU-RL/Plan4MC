import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
#from torch.distributions.normal import Normal
#from torch.distributions.categorical import Categorical
from collections import OrderedDict
import yaml


def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    #print(file_data)
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    #print(data)
    return data


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# to compute advantage functions
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# convert (3*W*H) [0,1] to (W*H*3) [0,255] numpy
def imgt2img(t):
    ret = np.transpose(np.asarray(t) * 255, [1,2,0]).astype(np.uint8)
    #print(ret)
    return ret


#Networks #####################################################################

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Conv2dLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
                 normalization='batch', nonlinear='relu'):
        if padding is None:
            padding = (kernel_size - 1) // 2

        bias = (normalization is None or normalization is False)

        modules = [nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )]

        if normalization is not None and normalization is not False:
            if normalization == 'batch':
                modules.append(nn.BatchNorm2d(num_features=out_channels))
            else:
                raise NotImplementedError(
                    'unsupported normalization layer: {0}'.format(normalization))

        if nonlinear is not None and nonlinear is not False:
            if nonlinear == 'relu':
                modules.append(nn.ReLU(inplace=True))
            elif nonlinear == 'leakyrelu':
                modules.append(nn.LeakyReLU(inplace=True))
            elif nonlinear == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise NotImplementedError(
                    'unsupported nonlinear activation: {0}'.format(nonlinear))

        super(Conv2dLayer, self).__init__(*modules)


from mineagent.actor.distribution import MultiCategorical
# for Minecraft, CNN actor of two-stage action
# input rgb image (3, 160, 256), output designed action (?, 3)
class CNNActor(nn.Module):
    
    def __init__(self, act1, act2=3):
        super().__init__()
        # 3*256*256 -> 128*4*4
        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(3, 32, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(32, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 128, 3, stride=2, normalization='batch', nonlinear='leakyrelu'))
        ]))

        self.action_dim = [act1, act2]
        self.act1_net = mlp([128*3*4, 128, 128, act1], nn.Tanh) # action1: categorical
        self.act2_net = mlp([128*3*4, 128, 128, act2], nn.Tanh) # action2: categorical
    
    '''
    def _distribution(self, obs):
        hid = self.encoder(obs)
        hid = hid.view(hid.size(0), -1)

        act1_logits = self.act1_net(hid)
        act2_logits = self.act2_net(hid)

        return Categorical(logits=act1_logits), Categorical(logits=act2_logits)
    
    def _log_prob_from_distribution(self, pi, act):
        # joint distribution: log p = log p1 + log p2
        #print(pi[0].log_prob(act[0]), pi[1].log_prob(act[1]).sum(axis=-1))
        return pi[0].log_prob(act[0]) + pi[1].log_prob(act[1])
    '''

    def forward(self, obs):
        hid = self.encoder(obs)
        hid = hid.view(hid.size(0), -1)
        return torch.cat([self.act1_net(hid), self.act2_net(hid)], dim=1)

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self.action_dim)


# value function
class CNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # 3*256*256 -> 128*4*4
        self.encoder = nn.Sequential(OrderedDict([
          ('conv0', Conv2dLayer(3, 32, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv1', Conv2dLayer(32, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv2', Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv3', Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv4', Conv2dLayer(64, 64, 3, stride=2, normalization='batch', nonlinear='leakyrelu')),
          ('conv5', Conv2dLayer(64, 128, 3, stride=2, normalization='batch', nonlinear='leakyrelu'))
        ]))
        self.mlp = mlp([128*3*4, 128, 128, 1], nn.Tanh)

    def forward(self, obs):
        hid = self.encoder(obs)
        hid = hid.view(hid.size(0), -1)
        v = self.mlp(hid)
        return torch.squeeze(v, -1)



from mineagent.batch import Batch
# Minecraft network
class CNNActorCritic(nn.Module):

    def __init__(
        self, 
        action_dim, 
        deterministic_eval: bool = False, # use stochastic in both exploration and test
    ):
        super().__init__()

        assert len(action_dim) == 2
        self.actor = CNNActor(action_dim[0], action_dim[1])
        self.critic  = CNNCritic()
        self._deterministic_eval = deterministic_eval
        self.dist_fn = self.actor.dist_fn


    # forward actor
    def forward(self, obs):
        logits = self.actor(obs)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            act = dist.mode()
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, dist=dist)

    # forward actor critic
    def forward_actor_critic(self, obs):
        logits = self.actor(obs)
        val = self.critic(obs)
        
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


# DQN #############################################
from mineclip.utils import build_mlp
from mineagent import features, SimpleFeatureFusion

class Qnet(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        *,
        action: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device,
    ):
        super().__init__()
        self.preprocess = preprocess_net
        self.net = build_mlp(
            input_dim=preprocess_net.output_dim,
            output_dim=action,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None,
        )
        self._action = action
        self._device = device

    def forward(self, x):
        y, _ = self.preprocess(x)
        return self.net(y)

class DQN:
    def __init__(self, agent_config, action_dim, device):
        feature_net_kwargs = agent_config['feature_net_kwargs']
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(features, cls)
            feature_net[k] = cls(**v, device=device)
        feature_fusion_kwargs = agent_config['feature_fusion']
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=device
        )
        #feature_net_v = copy.deepcopy(feature_net)  # actor and critic do not share
        # #feature_net finish
        self.dqn = Qnet(
            feature_net,
            action=action_dim,  #[12,3]
            device=device,
            **agent_config['actor'],
        ).to(device)
        self.dqn.eval()
        self.device = device
        self.action_dim = action_dim

    def take_action(self, obs, epsilon):
        if np.random.random() < epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            action = self.dqn(obs.obs).argmax().item()
        act = self.action_process(action)
        return act

    def action_process(self, act):
        action = torch.zeros((1,2), dtype=int)
        action[0][1] = act % 3
        action[0][0] = act // 3
        return action

    def load_model(self, pth):
        state_dict = torch.load(pth, map_location=self.device)
        self.dqn.load_state_dict(state_dict)
