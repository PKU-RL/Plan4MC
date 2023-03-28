import numpy as np
import torch
import random
import os
import imageio
import utils
import argparse
from skills import skills, skill_search, SkillsModel, convert_state_to_init_items

def main(args):
    # load task configs
    task_configs = utils.get_yaml_data(args.task_config_path)
    #print(task_conf)

    for k in task_configs.keys():
        task_conf = task_configs[k]
        init_items = {}
        if 'initial_inventory' in task_conf:
            init_items = task_conf['initial_inventory']
        #print(init_inv)

        target_name = task_conf['target_name']
        skill_sequence, init_items_miss = skill_search(target_name, init_items)
        if len(init_items_miss)>0:
            raise Exception('Cannot finish task because of missing initial items: {}'.format(init_items_miss))
        print('Planning for task: ', k)

        print('Initial planned skill sequence: {}, length: {}'.format(skill_sequence, len(skill_sequence)))
        skill_sequence_unique = list(set(skill_sequence))
        skill_sequence_unique.sort(key=skill_sequence.index)
        print('Involved skills: {}, length: {}'.format(skill_sequence_unique, len(skill_sequence_unique)))
        print('')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-config-path', type=str, default='envs/hard_task_conf.yaml')
    args = parser.parse_args()
    main(args)