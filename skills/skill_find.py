import pickle
import copy
import torch
import numpy as np
from docopt import docopt
from recurrent_ppo_truncated_bptt.model import ActorCriticModel
from recurrent_ppo_truncated_bptt.environments.navigation_env import MinecraftNav


class SkillFind:
    def __init__(self, device=torch.device('cuda:0')):
        self.device = device

    def execute(self, target, model_path, max_steps_high, max_steps_low, env, **kwargs):
        if target=='log':
            target = 'wood'
        elif target=='cobblestone':
            target = 'stone'
        env_high = MinecraftNav(max_steps=max_steps_high, usage='deploy', env=env, low_level_policy_type='dqn',
            device=self.device)
        state_dict, config = pickle.load(open(model_path, "rb"))
        model = ActorCriticModel(config, env_high.observation_space, (env_high.action_space.n,))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        # Init recurrent cell
        hxs, cxs = model.init_recurrent_cell_states(1, self.device)
        if config["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif config["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)

        obs = env_high.reset()
        done = False
        while not done:
            with torch.no_grad():
                policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)).float(), recurrent_cell, self.device, 1)
            action = policy.sample().cpu().numpy()
            obs, reward, done, info = env_high.step(int(action), target=target, max_steps_low=max_steps_low)
        
        if info['task_done'] or (not ('dis' in info)):
            return False, info['task_success'], info['task_done']

        success, r, done = env_high.reach(target, info)
        return success, r, done
