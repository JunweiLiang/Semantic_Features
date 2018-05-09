# coding=utf-8
# given a path full of npz file, for each npz file, averge the feature and return one single npy file.

import sys,os,argparse
import numpy as np
from tqdm import tqdm

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("npzlst",type=str)
	parser.add_argument("featpath",type=str,help="new path to store the averaged features")
	parser.add_argument("--l2norm",action="store_true")
	parser.add_argument("--ranknorm",action="store_true",help="rank normalization, only useful for using simple linear feature")

	return parser.parse_args()

def ranknorm(data):
	temp = data.argsort() # note equal item will have differnt ranks, but it is almost impossible for dense feature, so ignore it
	ranks = np.empty(len(data),"float32")
	ranks[temp] = np.arange(len(data))
	ranks+=1.0 # originaly the rank starts from zero
	data = ranks/len(ranks)
	return data

if __name__ == "__main__":
	args = get_args()

	assert not (args.l2norm and args.ranknorm)

	if not os.path.exists(args.featpath):
		os.makedirs(args.featpath)

	for npz in tqdm([line.strip() for line in open(args.npzlst,"r").readlines()]):

		filename = os.path.splitext(os.path.basename(npz))[0]

		data = np.load(npz)
		stacks = []
		for key in data.keys():
			stacks.append(data[key])
		feats = np.mean(stacks,axis=0)

		if args.l2norm:
			norm = np.linalg.norm(feats,2)
			feats/=norm

		if args.ranknorm:
			feats = ranknorm(feats)

		np.save(os.path.join(args.featpath,filename+".npy"),feats)