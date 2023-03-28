import yaml
import os
import torch
from tqdm import trange
import sys
import numpy as np
import utils
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

skills = utils.get_yaml_data('skills/skills.yaml')
#print(skills)

skill_names = list(skills.keys())
item_names = set()
for i, k in enumerate(skill_names):
	if skills[k]['consume'] is not None:
		for p in list(skills[k]['consume'].keys()):
			item_names.add(p)
	if skills[k]['require'] is not None:
		for p in list(skills[k]['require'].keys()):
			item_names.add(p)
	for p in skills[k]['equip']:
		item_names.add(p)
	if skills[k]['obtain'] is not None:
		for p in list(skills[k]['obtain'].keys()):
			item_names.add(p)
item_names = list(item_names)
#print(item_names)

name2id = {}
for i, k in enumerate(item_names):
	name2id[k] = i

'''
def vis_graph():
	G = nx.DiGraph()
	for i, k in enumerate(item_names):
		G.add_node(i, desc=k)

	for i, k in enumerate(skill_names):
		for q in skills[k]['obtain']:
			for p in skills[k]['consume']:
				G.add_edge(name2id[q], name2id[p])
			for p in skills[k]['require']:
				G.add_edge(name2id[q], name2id[p])
	pos = nx.shell_layout(G)
	node_labels = nx.get_node_attributes(G, 'desc')
	nx.draw_networkx(G, pos, with_labels=None)
	nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
	plt.show()
'''

possess = {}
def skill_search(target, init_items={}):
	results, init_items_miss = [], {}
	global possess
	possess = deepcopy(init_items)

	# tree search steps
	def recur(tgt, tgt_num):
		global possess
		# already done
		if tgt in list(possess.keys()) and possess[tgt]>=tgt_num:
			return
		# target cannot be produced in current skills
		if tgt not in skill_names:
			if tgt in init_items_miss:
				init_items_miss[tgt] += tgt_num
			else:
				init_items_miss[tgt] = tgt_num
			return

		tgt_sub, tgt_nearby = [], []
		#tgt_sub_num, tgt_nearby_num = [], []
		# first obtain consume items, then others, last nearby items.
		if skills[tgt]['consume'] is not None:
			for i in list(skills[tgt]['consume'].keys()): 
				if i.endswith('_nearby'):
					tgt_nearby.append((i, 0, skills[tgt]['consume'][i]))
				else:
					tgt_sub.append((i, 0, skills[tgt]['consume'][i]))
		if skills[tgt]['require'] is not None:
			for i in list(skills[tgt]['require'].keys()): 
				if i.endswith('_nearby'):
					tgt_nearby.append((i, 1, skills[tgt]['require'][i]))
				else:
					tgt_sub.append((i, 1, skills[tgt]['require'][i]))
		tgt_sub = tgt_sub + tgt_nearby

		#print(tgt, tgt_num, possess, tgt_sub)

		# sequentially solve the sub tasks
		for (k, c, n) in tgt_sub: # item, consume, number
			n_to_acquire = n
			# already have
			if k in list(possess.keys()):
				# have enough items
				if possess[k] >= n:
					if c==0: # consume
						possess[k]-=n
					continue
				# not enough
				else:
					n_to_acquire-=possess[k]
					if c==0: #consume
						possess[k]=0
			
			while n_to_acquire>0:
				recur(k, n) # do skill to obtain k
				# lose _nearby things after skill 0 (find skills)
				if (k in skill_names) and (skills[k]['skill_type']==0):
					for i in list(possess.keys()):
						if i.endswith('_nearby'):
							possess[i] = 0
			
				# maintain the obtained things, after skill execution
				if k not in skill_names:
					n_to_acquire -= n
					if c!=0: # preserve
						if k in list(possess.keys()):
							possess[k] += n
						else:
							possess[k] = n
				else:
					for obtain in list(skills[k]['obtain'].keys()):
						obtain_num = skills[k]['obtain'][obtain]
						if (obtain!=k) or c==1:
							if obtain in list(possess.keys()):
								possess[obtain] += obtain_num
							else:
								possess[obtain] = obtain_num
						elif n_to_acquire<obtain_num: # obtain==k and c==0, cost things
							if obtain in list(possess.keys()):
								possess[obtain] += (obtain_num-n_to_acquire)
							else:
								possess[obtain] = (obtain_num-n_to_acquire)
					#print(k, n_to_acquire, possess)
					n_to_acquire -= skills[k]['obtain'][k]

		#print(tgt, possess)
		results.append(tgt)
	
	# main search
	recur(target, 1)
	return results, init_items_miss


# after skill execution, update init_items for recomputing skill sequence
def convert_state_to_init_items(last_init_items, last_skill, last_skill_type, last_skill_done, 
	inventory_name, inventory_quantity):
	nearby_items = {}
	if last_skill_type==2: # craft skills do not lose nearby items
		for i in list(last_init_items.keys()):
			if i.endswith('_nearby'):
				nearby_items[i] = 1
	elif last_skill.endswith('_nearby'): # last skill obtains nearby items
		if last_skill_done:
			nearby_items[last_skill] = 1

	inventory_items = {}
	for n, q in zip(inventory_name, inventory_quantity):
		n_ = n.replace(' ', '_')
		if n_ in inventory_items:
			inventory_items[n_] += q
		else:
			inventory_items[n_] = q
	return dict(nearby_items, **inventory_items)

if __name__ == '__main__':
	#vis_graph()
	
	#for sk in skill_names:
	#	print(sk, skill_search(sk))
	my_tasks = [
		('crafting_table_nearby', {}),
		('furnace_nearby', {'crafting_table':1, 'cobblestone':8}),
		('milk_bucket', {'iron_ingot':3, 'crafting_table':1}),
		('wool', {'iron_ingot':2, 'crafting_table':1}),
		('beef', {}),
		('mutton', {}),
		('bed', {'shears':1, 'crafting_table':1}),
		('stick', {})
	]

	ssets = [set() for j in range(3)]
	for (k, i) in my_tasks:
		sks, _ =  skill_search(k, i)
		for s in sks:
			ssets[skills[s]['skill_type']].add(s)
		print(k, i, sks, _)
	#for i in range(3):
	#	print(ssets[i])

