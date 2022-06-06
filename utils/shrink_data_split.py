import torch
import os
import os.path as osp

def shrink(filename):
	dict = torch.load(split_dict, filename)
	train = data['train']
	shinked = train['train'][:1000]