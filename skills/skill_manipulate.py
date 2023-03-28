from envs.minecraft_hard_task import preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import pickle
import utils
import copy
import torch
import numpy as np

class SkillManipulate:
    def __init__(self, device=torch.device('cuda:0'), actor_out_dim=[12,3], agent_config_path='mineagent/conf.yaml'):
        agent_config = utils.get_yaml_data(agent_config_path)
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
        feature_net_v = copy.deepcopy(feature_net) # actor and critic do not share
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=actor_out_dim,
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        critic = Critic(
            feature_net_v,
            action_dim=None,
            device=device,
            **agent_config['actor'],
            activation='tanh'
        )
        self.mine_agent = MineAgent(
            actor=actor, 
            critic=critic,
            deterministic_eval=False
        ).to(device) # use the same stochastic policy in training and test
        self.mine_agent.eval()
        self.device=device

    def execute(self, target, model_path, max_steps, env, equip_list, **kwargs):
        state_dict = torch.load(model_path, map_location=self.device)
        self.mine_agent.load_state_dict(state_dict)

        # If equipment list is empty, we do not allow the use action
        allow_use = True if len(equip_list)>0 else False

        obs = env.obs
        for step in range(max_steps):
            batch = preprocess_obs(obs, self.device)
            with torch.no_grad():
                act = self.mine_agent(batch).act
            act = transform_action(act, allow_use)
            obs, r, done, _ = env.step(act)
            # detect skill done
            if env.reward_harvest(obs, target):
                return True, bool(r), done # skill done, task success, task done
            elif done:
                return False, bool(r), done # skill done, task success, task done
        return False, False, False # skill done, task success, task done
