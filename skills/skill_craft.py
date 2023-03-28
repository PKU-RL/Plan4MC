import numpy as np
from minedojo.sim.mc_meta.mc import ALL_CRAFT_SMELT_ITEMS

class SkillCraft:
    def __init__(self):
        self.item2id = {}
        for i, n in enumerate(ALL_CRAFT_SMELT_ITEMS):
            self.item2id[n] = i
        #print(self.item2id)

    def execute(self, target, env):
        act = env.base_env.action_space.no_op()
        if not (target in self.item2id):
            print('Warning: target {} is not in the crafting list'.format(target))
            return False, False, False # skill done, task success, task done

        target_id = self.item2id[target]
        act[5] = 4
        act[6] = target_id
        obs, r, done, _ = env.step(act)

        if env.reward_harvest(obs, target):
            return True, bool(r), done # skill done, task success, task done
        else:
            return False, bool(r), done # skill done, task success, task done

if __name__=='__main__':
    s = SkillCraft()
    print(s.item2id)