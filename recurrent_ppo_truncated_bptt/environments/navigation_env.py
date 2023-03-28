import gym
import numpy as np
import time
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import sys
sys.path.append('..')
from envs.minecraft_nav import MinecraftNavEnv, MinecraftNavTestEnv, transform_action, preprocess_obs
import utils
from mineclip_official import build_pretrain_model
import torch

def naive_policy(obs):
    goal_emb = obs['goal_emb']
    yaw = goal_emb[2:4]
    dr = goal_emb[4:6]
    dr = np.array([dr[1], -dr[0]])
    dr /= np.linalg.norm(dr)

    act = [0,0,0,12,12,0,0,0]
    # correct the pitch direction
    pitch = obs["location_stats"]["pitch"]
    if pitch>20:
        act[3] = 10
        return act
    elif pitch<-20:
        act[3] = 14
        return act
    # the direction is correct: forward or jump
    if np.dot(dr, yaw)>=np.cos(np.deg2rad(20)):
        if np.random.rand()<0.8:
            act[0] = 1
        else:
            act[2] = 1
        return act
    # should turn left
    if yaw[1]*dr[0]>=yaw[0]*dr[1]:
        act[4] = 10
    # turn right
    else:
        act[4] = 14
    #print('yaw {} goal_pos {} dir {} act {}'.format(np.rad2deg(np.arccos(yaw[0])), goal_emb[4:6], np.rad2deg(np.arccos(dr[0])), act))
    return act

def dqn_policy(obs, dqn, epsilon=0.05):
    batch = preprocess_obs(obs, dqn.device)
    act = dqn.take_action(batch, epsilon)
    act = transform_action(act)
    return act


class MinecraftNav:
    def __init__(self, seed=7, max_steps=40, usage='train', env=None, low_level_policy_type='dqn', device=None):
        self.xlim = [-5, 5]
        self.zlim = [-5, 5]
        self.xrange, self.zrange = 11, 11

        self.low_level_policy_type=low_level_policy_type
        if low_level_policy_type=='naive':
            self.low_level_policy = naive_policy
            self.clip_model = None
            self.device = None
        elif low_level_policy_type=='dqn':
            if usage != 'deploy':
                agent_config_path = '../mineagent/conf_goal_based_agent.yaml'
                dqn_model_path = 'environments/model_770.pth'
            else:
                agent_config_path = 'mineagent/conf_goal_based_agent.yaml'
                dqn_model_path = 'skills/models/find_low.pth'

            agent_config = utils.get_yaml_data(agent_config_path)
            self.dqn = utils.DQN(agent_config, 12*3, device)
            self.dqn.load_model(dqn_model_path)
            self.low_level_policy = lambda x: dqn_policy(x, self.dqn)
            self.device = device
            self.clip_model = None
            if usage != 'deploy':
                clip_config = utils.get_yaml_data('../mineclip_official/config.yml')
                self.clip_model = build_pretrain_model(
                    image_config=clip_config['image_config'],
                    text_config=clip_config['text_config'],
                    temporal_config=clip_config['temporal_config'],
                    adapter_config=clip_config['adaptor_config'],
                    state_dict=torch.load('../mineclip_official/adjust.pth')
                ).to(device)
                self.clip_model.eval()
                print('MineCLIP model loaded.')
        else:
            raise NotImplementedError

        self.usage = usage
        # training environment: any plains
        if usage == 'train': 
            self.env = MinecraftNavEnv(
                    image_size=(160, 256),
                    clip_model=self.clip_model, 
                    device=self.device,
                    seed=seed,
                    biome='plains'
                )
        # test navigation environment: plains with cow and sheep
        elif usage == 'test':
            self.env = MinecraftNavTestEnv(
                    image_size=(160, 256),
                    clip_model=self.clip_model, 
                    device=self.device,
                    seed=seed,
                    biome='plains'
                )
        # deploy: use the existing env
        elif usage == 'deploy':
            self.env = env
        else:
            raise NotImplementedError
        self.max_steps=max_steps

    @property
    def observation_space(self):
        return Box(low=-100, high=100, shape=(2,), dtype=np.float32)
    
    @property
    def action_space(self):
        return Discrete(4)

    def _envpos2gridpos(self, pos):
        return np.round((pos - self.env_pos_begin)/10.).astype(int)

    def reset(self):
        #if hasattr(self, 'visit'):
        #    print(self.visit)
            
        self.visit = np.zeros((self.xrange, self.zrange), dtype=int)
        #self.pos = [0,0]
        self.visit[self.xrange//2,self.zrange//2]=1

        self.n_steps=0

        self._rewards = []

        if self.usage == 'deploy':
            self.obs_env = self.env.obs
        else:
            self.obs_env = self.env.reset()
        self.env_pos_begin = self.obs_env['location_stats']['pos']
        self.env_pos_begin = np.array([self.env_pos_begin[0], self.env_pos_begin[2]])
        # high-level policy observes the relative env pos to the start pos
        self.visited_pos = [np.array([0.,0.])]
        return np.array([0.,0.])

    def step(self, action, target=None, max_steps_low=None):
        assert action>=0 and action<4

        if action==0:
            act = [-1.,0.]
        elif action==1:
            act = [1.,0.]
        elif action==2:
            act = [0.,1.]
        else:
            act = [0.,-1.]
        self.env.set_goal(self.obs_env['location_stats']['pos'], act)

        reward_state = 0 # state count reward
        reward_invalid = 0 # if reach some position out of grid, give a penalty
        # execute low level policy
        done=False
        agent_dead = False
        tgt_found = False # for test mode
        n_steps_low = 0
        while not done:
            self.env.add_goal_to_obs(self.obs_env)
            a_env = self.low_level_policy(self.obs_env)
            next_o, r, done, _ = self.env.step(a_env)
            self.obs_env = next_o

            # update state visitation
            pos = next_o['location_stats']['pos']
            pos = np.array([pos[0], pos[2]])
            self.env_pos = np.array(pos)
            pos = self._envpos2gridpos(pos)
            if pos[0]<self.xlim[0] or pos[0]>self.xlim[1] or pos[1]<self.zlim[0] or pos[1]>self.zlim[1]:
                reward_invalid = -10
            else:
                xx = pos[0] + self.xrange//2
                zz = pos[1] + self.zrange//2
                if not self.visit[xx,zz]:
                    self.visit[xx,zz]=1
                    reward_state += 1
            if next_o['life_stats']['life']==0:
                agent_dead=True
                break
            # search for the target
            if self.usage == 'test' or self.usage == 'deploy':
                tgt_find, tgt_info = self.env.target_in_sight(next_o, target)
                if tgt_find:
                    tgt_found = True
                    break
            # for deploy case
            n_steps_low+=1
            if self.usage=='deploy' and n_steps_low>=max_steps_low:
                break

        if self.usage!='deploy':
            self.env.reset(reset_env=False)
        elif done: # deploy and env done
            return np.array(self.env_pos-self.env_pos_begin), 0, True, {"skill_done": tgt_found, "task_success": bool(r), "task_done": True}

        self.n_steps += 1
        done = False
        if self.n_steps >= self.max_steps or agent_dead or tgt_found:
            done = True

        '''
        ent = self.entities[xx,zz]
        if ent!=-1 and (not self.vis_entity[ent]):
            self.vis_entity[ent] = 1
            reward_entity=5
        else:
            reward_entity=0
        '''

        reward = reward_invalid + reward_state #+ reward_entity
        #print('rewards:', reward_invalid, reward_state)
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
            if tgt_found:
                info.update(tgt_info)
            if self.usage=='deploy':
                info.update({"skill_done": tgt_found, "task_success": False, "task_done": False})
        else:
            info = None

        self.visited_pos.append(np.array(self.env_pos-self.env_pos_begin))
        # high-level policy observes the relative env pos to the start pos
        return np.array(self.env_pos-self.env_pos_begin), reward, done, info


    # test and deploy mode: after find the target, reach it with goal-based policy
    def reach(self, target, info, max_dis=3, max_steps=200):
        assert ('dis' in info)
        assert (self.usage == 'test' or self.usage == 'deploy')
        dis, yaw = info['dis'], info['yaw']
        yaw = np.deg2rad(yaw)
        self.env.set_goal(self.obs_env['location_stats']['pos'], [np.cos(yaw), np.sin(yaw)])

        success = False
        step_cnt = 0
        for i in range(max_steps):
            self.env.add_goal_to_obs(self.obs_env)
            #print(i,'goalpos {}, pos {}'.format(self.env.goal_pos, self.obs_env['location_stats']['pos']))
            a_env = naive_policy(self.obs_env) #self.low_level_policy(self.obs_env)
            next_o, r, done, _ = self.env.step(a_env)
            self.obs_env = next_o

            if next_o['life_stats']['life']==0 or (self.usage=='deploy' and done):
                break
            # search for the target
            tgt_find, tgt_info = self.env.target_in_sight(next_o, target)
            if tgt_find and tgt_info['dis']<=max_dis:
                success=True
                break

            # reset goal after several steps
            if step_cnt >= 50 and tgt_find and tgt_info['dis']<dis:
                dis = tgt_info['dis']
                step_cnt = 0
                yaw = np.deg2rad(tgt_info['yaw'])
                self.env.set_goal(self.obs_env['location_stats']['pos'], [np.cos(yaw), np.sin(yaw)])
            step_cnt += 1

        # log position
        pos = self.obs_env['location_stats']['pos']
        pos = np.array([pos[0], pos[2]])
        self.env_pos = np.array(pos)
        self.visited_pos.append(np.array(self.env_pos-self.env_pos_begin))

        if self.usage=='deploy':
            return success, bool(r), done # skill done, task success, task done
        return success


    def render(self):
        print(self.visit)
        #time.sleep(0.033)

    def close(self):
        pass
        #self._env.close()
