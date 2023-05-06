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
    compass=torch.as_tensor([np.concatenate([np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)])], device=device)
    obs_ = {
        "compass": compass,
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
        #"goal": torch.as_tensor(obs["goal_emb"], dtype=torch.float, device=device).view(B, 6), 
    }
    #print(obs_["prev_action"])
    #print(obs_["compass"], yaw_, pitch_)
    #print(obs_["goal"])

    #print(Batch(obs=obs_))
    return Batch(obs=obs_)



# Map mine-agent action to env action.
# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 4 camera actions
def transform_action(act, allow_use=True):
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
    if act2 == 1 and allow_use: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)


SUBGOAL_DISTANCE=10
# environment for hard harvest tasks
# support 3 types of skills
class MinecraftHardHarvestEnv:
    def __init__(self, image_size=(160, 256), seed=0, biome='plains', clip_model=None, device=None, 
        target_name='log', target_quantity=1, save_rgb=False, max_steps=3000, **kwargs):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.biome = biome
        self.max_step = max_steps
        self.cur_step = 0
        self.seed = seed
        self.image_size = image_size
        self.kwargs = kwargs # kwargs should contain: initial inventory, initial mobs
        self.target_name = target_name
        self.target_quantity = target_quantity
        self.remake_env()
        #self._first_reset = True
        #self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
        self.clip_model = clip_model # use mineclip model to precompute embeddings
        self.device = device
        self.save_rgb = save_rgb

    def __del__(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

    def remake_env(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

        if self.target_name.endswith('_nearby'):
            self.target_item_name = self.target_name[:-7]
        else:
            self.target_item_name = self.target_name

        self.base_env = minedojo.make(
            task_id="harvest", 
            image_size=self.image_size, 
            target_names=self.target_item_name,
            target_quantities=self.target_quantity,
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
            # teleport agent to a new place when reset
            fast_reset=True,
            fast_reset_random_teleport_range_low=0,
            fast_reset_random_teleport_range_high=500,
            # spawn initial mobs
            #initial_mobs=['cow']*3 + ['sheep']*3,
            #initial_mob_spawn_range_low=(-30, 1, -30),
            #initial_mob_spawn_range_high=(30, 1, 30),
            **self.kwargs)
        #self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')

    def reset(self):
        self.cur_step = 0
        self.prev_action = self.base_env.action_space.no_op()
        self.base_env.reset(move_flag=True) # reset after random teleport, spawn mobs nearby
        self.base_env.unwrapped.set_time(6000)
        self.base_env.unwrapped.set_weather("clear")
        # make agent fall onto the ground after teleport
        for i in range(4):
            obs, _,_,_ = self.base_env.step(self.base_env.action_space.no_op())

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        if self.save_rgb:
            self.rgb_list = [np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)]
            self.action_list = []
        self.obs = obs
        self.last_obs = obs
        return obs

    def step(self, act):
        obs, reward, done, info = self.base_env.step(act)
        if self.target_name.endswith('_nearby'):
            reward = self.reward_harvest(obs, self.target_name)
            done = True if reward>0 else False

        if obs['life_stats']['life']==0:
            done=True
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done=True
        if reward>0:
            reward=1
            done=True
        
        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        self.prev_action = act # save the previous action for the agent's observation
        self.last_obs = self.obs
        self.obs = obs
        if self.save_rgb:
            self.rgb_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
            self.action_list.append(np.asarray(act))
        return  obs, reward, done, info

    # for Find skill: detect target items
    def target_in_sight(self, obs, target, max_dis=20):
        if target in ['wood', 'stone']:
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

    '''
    for goal-based find skill
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


    # compute harvest reward under different cases
    def reward_harvest(self, obs, target_name, target_quantity=1, incremental=True):
        # target nearby
        if target_name.endswith('_nearby'):
            target_item_name = target_name[:-7]
            if target_item_name == 'furnace':
                return int(obs['nearby_tools']['furnace'])
            elif target_item_name == 'crafting_table':
                return int(obs['nearby_tools']['table'])
            else:
                if target_item_name == 'log':
                    target_item_name = 'wood'
                elif target_item_name == 'cobblestone':
                    target_item_name = 'stone'
                find, info = self.target_in_sight(obs, target_item_name)
                if find and info['dis']<=3:
                    return 1
                else:
                    return 0
        # target in inventory
        else:
            names, nums = obs['inventory']['name'], obs['inventory']['quantity']
            #print('inventory:',names)
            idxs = np.where(names==target_name.replace('_',' '))[0]
            if len(idxs)==0:
                return 0
            else:
                num_cur = np.sum(nums[idxs])
                num_last = 0
                if incremental:
                    names, nums = self.last_obs['inventory']['name'], self.last_obs['inventory']['quantity']
                    idxs = np.where(names==target_name.replace('_',' '))[0]
                    if len(idxs)>0:
                        num_last = np.sum(nums[idxs])
                if num_cur-num_last>=target_quantity:
                    return 1
                else:
                    return 0