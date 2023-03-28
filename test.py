import numpy as np
import torch
import random
import os
import imageio
import utils
import argparse
from mineclip_official import build_pretrain_model
from envs.minecraft_hard_task import MinecraftHardHarvestEnv
from skills import skills, skill_search, SkillsModel, convert_state_to_init_items
from minedojo.sim import InventoryItem
import matplotlib.pyplot as plt

def main(args):
    # save path
    save_dir = args.save_path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.task)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print('Running on device: ', device)

    # seed control
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load clip model
    clip_config = utils.get_yaml_data(args.clip_config_path)
    model_clip = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load(args.clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded from:', args.clip_model_path)

    # load task configs
    task_conf = utils.get_yaml_data(args.task_config_path)[args.task]
    #print(task_conf)
    init_items = {}
    if 'initial_inventory' in task_conf:
        init_items = task_conf['initial_inventory']
        init_inv = [InventoryItem(slot=i, name=k, variant=None, quantity=task_conf['initial_inventory'][k]) 
        for i,k in enumerate(list(task_conf['initial_inventory'].keys()))]
        task_conf['initial_inventory'] = init_inv
    #print(init_inv)

    # ablation for max steps
    if args.shorter_episode:
        task_conf['max_steps'] = task_conf['max_steps']//2

    print('task configs', task_conf)
    # Instantiate environment
    env = MinecraftHardHarvestEnv(
        image_size=(160,256),
        seed=seed,
        clip_model=model_clip,
        device=device,
        save_rgb=args.save_gif,
        **task_conf
        )

    # load skills
    skills_model = SkillsModel(device=device, path=args.skills_model_config_path)

    # run test
    target_name = task_conf['target_name']
    skill_sequence, init_items_miss = skill_search(target_name, init_items)
    if len(init_items_miss)>0:
        raise Exception('Cannot finish task because of missing initial items: {}'.format(init_items_miss))
    print('Task {} decomposed into skill sequence: {}'.format(args.task, skill_sequence))

    skill_success_cnt = np.zeros(len(skill_sequence))
    print('Initial skill sequence: {}, length: {}'.format(skill_sequence, len(skill_sequence)))
    skill_sequence_unique = list(set(skill_sequence))
    skill_sequence_unique.sort(key=skill_sequence.index)
    skill_success_cnt_unique = np.zeros(len(skill_sequence_unique))
    print('Unique skill list: {}, length: {}'.format(skill_sequence_unique, len(skill_sequence_unique)))
    test_success_rate = 0

    for ep in range(args.test_episode):
        env.reset()
        episode_snapshots = [('begin', np.transpose(env.obs['rgb'], [1,2,0]).astype(np.uint8))]

        # sequentially solve the initial computed skills. 
        if not args.progressive_search:
            assert args.no_find_skill==0
            assert args.shorter_episode==0
            episode_skill_success_unique = np.zeros(len(skill_sequence_unique))
            for i_sk, sk in enumerate(skill_sequence):
                print('executing skill:',sk)
                skill_done, task_success, task_done = skills_model.execute(skill_name=sk, skill_info=skills[sk], env=env)
                if skill_done or task_success:
                    skill_success_cnt[i_sk]+=1
                    episode_skill_success_unique[skill_sequence_unique.index(sk)]=1
                    episode_snapshots.append((sk, np.transpose(env.obs['rgb'], [1,2,0]).astype(np.uint8)))
                if (not skill_done) or task_done:
                    break
            #print(skill_success_cnt)
            print('skill done {}, task success {}, task done {}'.format(skill_done, task_success, task_done))
            skill_success_cnt_unique += episode_skill_success_unique
        # update the future skill sequence after each skill.
        else:
            episode_skill_success = np.zeros(len(skill_sequence))
            episode_skill_success_unique = np.zeros(len(skill_sequence_unique))
            episode_skill_idx = 0
            skill_next = skill_sequence[0]
            # ablation: skip find skills
            if args.no_find_skill and skills[skill_next]['skill_type']==0:
                skill_next = skill_sequence[1]
                assert skills[skill_next]['skill_type']!=0
            init_items_next = init_items
            while True:
                print('executing skill:',skill_next)
                skill_done, task_success, task_done = skills_model.execute(skill_name=skill_next, skill_info=skills[skill_next], env=env)
                if skill_done or task_success:
                    if skill_next in skill_sequence[episode_skill_idx:]:
                        episode_skill_idx += skill_sequence[episode_skill_idx:].index(skill_next)
                        episode_skill_success[episode_skill_idx] = 1
                        episode_skill_idx += 1
                    if skill_next in skill_sequence_unique:
                        episode_skill_success_unique[skill_sequence_unique.index(skill_next)]=1
                    episode_snapshots.append((skill_next, np.transpose(env.obs['rgb'], [1,2,0]).astype(np.uint8)))
                if task_done:
                    break
                init_items_next = convert_state_to_init_items(init_items_next, skill_next, skills[skill_next]['skill_type'],
                    skill_done, env.obs['inventory']['name'], env.obs['inventory']['quantity'])
                skill_sequence_next, items_miss = skill_search(target_name, init_items_next)
                skill_next = skill_sequence_next[0]
                print('recomputed skill sequence:', skill_sequence_next)
                # ablation: skip find skills
                if args.no_find_skill and skills[skill_next]['skill_type']==0:
                    skill_next = skill_sequence_next[1]
                    assert skills[skill_next]['skill_type']!=0
                if len(items_miss)>0:
                    print('cannot execute some skills:', items_miss)
                    break
            print('task done {}'.format(task_done))
            skill_success_cnt += episode_skill_success
            skill_success_cnt_unique += episode_skill_success_unique
            print('episode skill success', episode_skill_success)
        
        if task_success:
            test_success_rate += 1
        # save gif
        if args.save_gif:
            imageio.mimsave(os.path.join(save_dir,'episode{}_success{}.gif'.format(ep,int(task_success))), env.rgb_list, duration=0.1)
        # save snapshots
        save_dir_snapshots = os.path.join(save_dir, 'episode{}_success{}'.format(ep,int(task_success)))
        if not os.path.exists(save_dir_snapshots):
            os.mkdir(save_dir_snapshots)
        for i, (sk, im) in enumerate(episode_snapshots):
            imageio.imsave(os.path.join(save_dir_snapshots, '{}_{}.png'.format(i,sk)), im)
        print()

    # draw skill success figure
    plt.bar([i for i in range(len(skill_sequence))], skill_success_cnt/args.test_episode, align="center", color="b",
        tick_label=skill_sequence)
    plt.ylabel('success rate')
    plt.savefig(os.path.join(save_dir,'success_skills.png'))
    plt.cla()
    plt.bar([i for i in range(len(skill_sequence_unique))], skill_success_cnt_unique/args.test_episode, align="center", color="b",
        tick_label=skill_sequence_unique)
    plt.ylabel('success rate')
    plt.savefig(os.path.join(save_dir,'success_skills_unique.png'))
    plt.cla()
    print('success_skills', skill_success_cnt/args.test_episode, 'success_skills_unique', skill_success_cnt_unique/args.test_episode)

    test_success_rate /= args.test_episode
    print('success rate:', test_success_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='harvest_milk_with_crafting_table_and_iron_ingot')
    parser.add_argument('--progressive-search', type=int, default=1) # set to 0 for zero-shot planning
    parser.add_argument('--shorter-episode', type=int, default=0) # ablation for using 1/2 episode steps?
    parser.add_argument('--no-find-skill', type=int, default=0) # ablation without find-skill?
    parser.add_argument('--test-episode', type=int, default=30) # number of test episodes
    parser.add_argument('--seed', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--save-gif', type=int, default=0) # save whole gifs?
    parser.add_argument('--save-path', type=str, default='test_hard_tasks')
    parser.add_argument('--clip-config-path', type=str, default='mineclip_official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineclip_official/attn.pth')
    parser.add_argument('--task-config-path', type=str, default='envs/hard_task_conf.yaml')
    parser.add_argument('--skills-model-config-path', type=str, default='skills/load_skills.yaml')
    args = parser.parse_args()
    main(args)