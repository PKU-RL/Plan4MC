#from environments.cartpole_env import CartPole
#from environments.minigrid_env import Minigrid
#from environments.poc_memory_env import PocMemoryEnv
from environments.navigation_env import MinecraftNav

def create_env(env_name:str, **kwargs):
    """Initializes an environment based on the provided environment name.
    
    Args:
        env_name {str}: Name of the to be instantiated environment

    Returns:
        {env}: Returns the selected environment instance.
    """
    if env_name == "MinecraftNav":
        return MinecraftNav(**kwargs)
    '''
    if env_name == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True)
    if env_name == "CartPole":
        return CartPole(mask_velocity=False)
    if env_name == "CartPoleMasked":
        return CartPole(mask_velocity=True)
    if env_name == "Minigrid":
        return Minigrid()
    '''

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Args:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)




# DQN #############################################
from mineclip.utils import build_mlp
from mineagent import features, SimpleFeatureFusion
import yaml
import torch
import torch.nn as nn
import numpy as np

def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    #print(file_data)
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    #print(data)
    return data

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
