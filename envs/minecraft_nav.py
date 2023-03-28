import minedojo
import sys
#import imageio
import numpy as np
import time

from mineagent.batch import Batch
import torch
from mineclip_official import torch_normalize
from mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP

def preprocess_obs(obs, device):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1

    def cvt_voxels(vox):
        ret = np.zeros(3*3*3, dtype=np.long)
        for i, v in enumerate(vox.reshape(3*3*3)):
            if v in VOXEL_BLOCK_NAME_MAP:
                ret[i] = VOXEL_BLOCK_NAME_MAP[v]
        return ret

    # I consider the move and functional action only, because the camera space is too large?
    # construct a 3*3*4*3 action embedding
    def cvt_action(act):
        if act[5]<=1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5]==3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            #raise Exception('Action[5] should be 0,1,3')
            return 0

    yaw_ = np.deg2rad(obs["location_stats"]["yaw"])
    pitch_ = np.deg2rad(obs["location_stats"]["pitch"])
    obs_ = {
        "compass": torch.as_tensor([np.concatenate([np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)])], device=device),
        "gps": torch.as_tensor([obs["location_stats"]["pos"]], device=device),
        "voxels": torch.as_tensor(
            [cvt_voxels(obs["voxels"]["block_name"])], dtype=torch.long, device=device
        ),
        "biome_id": torch.tensor(
            [int(obs["location_stats"]["biome_id"])], dtype=torch.long, device=device
        ),
        "prev_action": torch.tensor(
            [cvt_action(obs["prev_action"])], dtype=torch.long, device=device
        ),
        "prompt": torch.as_tensor(obs["rgb_emb"], device=device).view(B, 512), 
        # this is actually the image embedding, not prompt embedding (for single task)
        "goal": torch.as_tensor(obs["goal_emb"], dtype=torch.float, device=device).view(B, 6), 
    }
    #print(obs_["prev_action"])
    #print(obs_["compass"], yaw_, pitch_)
    #print(obs_["goal"])

    #print(Batch(obs=obs_))
    return Batch(obs=obs_)



# Map agent action to env action.
# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 4 camera actions
def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8: # camera pitch 10
        action[3] = 10
    elif act1 == 9: # camera pitch 14
        action[3] = 14
    elif act1 == 10: # camera yaw 10
        action[4] = 10
    elif act1 == 11: # camera yaw 14
        action[4] = 14

    assert act2 < 3
    '''
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    '''
    # for find skill, ban the use action
    if act2 == 2: #attack
        action[5] = 3
    return action #(8)


'''
# [6, 3] action space, 1 choice among walk, jump and camera
# preserve forward, jump and 4 camera actions. Discard no-op.
def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 6
    #if act1 == 0: # no op
    #    action = action
    if act1 == 0: # forward
        action[0] = 1
    elif act1 == 1: # jump
        action[2] = 1
    elif act1 == 2: # camera pitch 10
        action[3] = 10
    elif act1 == 3: # camera pitch 14
        action[3] = 14
    elif act1 == 4: # camera yaw 10
        action[4] = 10
    elif act1 == 5: # camera yaw 14
        action[4] = 14

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)
'''


# environment for training find skill
# reward is computed conditioned on the goal
SUBGOAL_DISTANCE=10 # subgoal distance
SUBGOAL_STEPS=50 # steps to reach a subgoal

from collections import deque
class MinecraftNavEnv:

    def __init__(self, image_size=(160, 256), seed=0, biome='plains', 
        clip_model=None, device=None,  **kwargs):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.biome = biome
        self.max_step = SUBGOAL_STEPS
        self.cur_step = 0
        self.seed = seed
        self.image_size = image_size
        self.kwargs = kwargs
        self.remake_env()
        #self._first_reset = True
        #self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
        self.clip_model = clip_model # use mineclip model to precompute embeddings
        self.device = device

    def __del__(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

    '''
    pos: (x,y,z) current position
    g: (cos t, sin t) target yaw direction
    '''
    def set_goal(self, pos, g=None):
        if g is None:
            #g = 2*np.pi*np.random.rand()
            g = 0.5*np.pi*np.random.randint(0, 4) # simpler: 4 goals
            g = [np.cos(g), np.sin(g)]

        self.goal = np.array(g)
        self.init_pos = np.array([pos[0], pos[2]])
        # goal position (x',z')
        self.goal_pos = np.array([pos[0]-SUBGOAL_DISTANCE*g[1], pos[2]+SUBGOAL_DISTANCE*g[0]])
        self.prev_distance = np.linalg.norm(self.init_pos-self.goal_pos)
        return g

    def add_goal_to_obs(self, obs):
        yaw = np.deg2rad(obs["location_stats"]["yaw"])
        yaw = np.concatenate([np.cos(yaw), np.sin(yaw)])
        pos = obs['location_stats']['pos']
        pos = np.array([pos[0], pos[2]]) # [x,z]
        obs['goal_emb'] = np.concatenate([self.goal, yaw, self.goal_pos-pos])

    def remake_env(self):
        '''
        call this to reset all the blocks and trees
        should modify line 479 in minedojo/tasks/__init__.py, deep copy the task spec dict:
            import deepcopy
            task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
        '''
        if hasattr(self, 'base_env'):
            self.base_env.close()
        self.base_env = minedojo.make(
            task_id="harvest", 
            image_size=self.image_size, 
            target_names='log',
            target_quantities=64,
            reward_weights=1,
            world_seed=self.seed, 
            seed=self.seed,
            specified_biome=self.biome, 
            use_voxel=True,
            voxel_size={ 'xmin': -1, 'ymin': -1, 'zmin': -1, 'xmax': 1, 'ymax': 1, 'zmax': 1 },
            **self.kwargs)
        #self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')


    def reset(self, reset_env=True, random_teleport=True):
        self.cur_step = 0
        # reset after finishing subgoal
        if not reset_env:
            return

        # big reset
        #if not self._first_reset:
            #for cmd in self._reset_cmds:
            #    self.base_env.unwrapped.execute_cmd(cmd)
        #self._first_reset = False
        self.prev_action = self.base_env.action_space.no_op()
        obs = self.base_env.reset()
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")
        #print(obs['location_stats']['pos'])

        # random teleport agent
        if random_teleport:
            self.base_env.random_teleport(200)
            self.base_env.step(self.base_env.action_space.no_op())
            obs, _, _, _ = self.base_env.step(self.base_env.action_space.no_op())
            # I find that position in obs is updated after 2 env.step
            #print(obs['location_stats']['pos'])
        
        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        return obs

    def step(self, act):
        obs, _, done, info = self.base_env.step(act)
        #agent_dead = done
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True
        
        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        self.prev_action = act # save the previous action for the agent's observation

        # compute navigation reward
        yaw = np.deg2rad(obs['location_stats']['yaw'][0])
        reward_yaw = np.cos(yaw) * self.goal[0] + np.sin(yaw) * self.goal[1] # [-1,1]
        pitch = np.deg2rad(obs['location_stats']['pitch'][0])
        reward_pitch = np.cos(pitch) # [0,1]
        pos = obs['location_stats']['pos']
        pos = np.array([pos[0], pos[2]]) # [x,z]
        dis = np.linalg.norm(pos-self.goal_pos) # generally [-10,0]
        reward_dis = self.prev_distance - dis # [-0.2, 0.2]
        reward = reward_yaw + reward_pitch + reward_dis*10 # [-3,4]
        self.prev_distance = dis
        #reward = reward_dis
        obs['reward_yaw'] = reward_yaw
        obs['reward_dis'] = reward_dis
        obs['reward_pitch'] = reward_pitch

        #info['agent_dead'] = agent_dead
        return  obs, reward, done, info


# testing environment for navigation
# add lidar to detect targets
class MinecraftNavTestEnv(MinecraftNavEnv):
    def __init__(self, image_size=(160, 256), seed=0, biome='plains', 
        clip_model=None, device=None, **kwargs):
        super().__init__(image_size=image_size, seed=seed, biome=biome, 
            clip_model=clip_model, device=device,  **kwargs)

    def remake_env(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()
        self.base_env = minedojo.make(
            task_id="harvest", 
            image_size=self.image_size, 
            target_names='log',
            target_quantities=64,
            reward_weights=1,
            world_seed=self.seed, 
            seed=self.seed,
            specified_biome=self.biome, 
            use_voxel=True,
            voxel_size={ 'xmin': -1, 'ymin': -1, 'zmin': -1, 'xmax': 1, 'ymax': 1, 'zmax': 1 },
            use_lidar=True,
            lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 99)
                for pitch in np.arange(-30, 30, 5)
                for yaw in np.arange(-45, 45, 5)
            ],
            # spawn initial mobs
            initial_mobs=['cow']*3 + ['sheep']*3,
            initial_mob_spawn_range_low=(-30, 1, -30),
            initial_mob_spawn_range_high=(30, 1, 30),
            # teleport agent when reset
            fast_reset=True,
            fast_reset_random_teleport_range_low=0,
            fast_reset_random_teleport_range_high=200,
            **self.kwargs)
        #self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')

    def reset(self, reset_env=True):
        self.cur_step = 0
        # reset after finishing subgoal
        if not reset_env:
            return
        # big reset
        self.prev_action = self.base_env.action_space.no_op()
        self.base_env.reset(move_flag=True) # reset after random teleport, spawn mobs nearby
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")
        # make agent fall onto the ground after teleport
        for i in range(4):
            obs, _, _, _ = self.base_env.step(self.base_env.action_space.no_op())
        self.total_steps = 0

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action
        return obs

    def step(self, act):
        self.total_steps += 1
        return super().step(act)

    # detect target items
    def target_in_sight(self, obs, target, max_dis=20):
        if target in ['wood']:
            target_type = 'block'
        elif target in ['cow', 'sheep']:
            target_type = 'entity'
        else:
            raise NotImplementedError

        #print(np.rad2deg(obs['rays']['ray_yaw']), obs['rays'][target_type+'_name'])
        names, distances = obs['rays'][target_type+'_name'], obs['rays'][target_type+'_distance']
        idxs = np.where(names==target)[0]
        if len(idxs)==0:
            return False, None
        idx = idxs[np.argmin(distances[idxs])]
        dis = distances[idx]
        if dis>max_dis:
            return False, None
        yaw_relative = -np.rad2deg(obs['rays']['ray_yaw'][idx]) # minedojo bug! yaw in lidar is opposite.
        yaw = obs["location_stats"]["yaw"][0] + yaw_relative
        pos = obs['location_stats']['pos']
        dr = [np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))]
        target_pos = np.array([pos[0]-dis*dr[1], pos[2]+dis*dr[0]])
        return True, {'dis':dis, 'yaw':yaw, 'yaw_relative':yaw_relative, 'target_pos':target_pos}


if __name__ == '__main__':
    #print(minedojo.ALL_TASKS_SPECS)
    env = MinecraftNavTestEnv()
    #reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
    obs = env.reset()
    #print(obs.shape, obs.dtype)
    for t in range (1000):
        act = env.base_env.action_space.no_op()
        act[4] = 13
        next_obs, r, done, info = env.step(act)
    