# coding=utf-8
# giving framelst for all the videos, each framepath has all the jpg,  this extract feature for all the jpg and averge them

# this code assumes the number of jpgs in one path is small

import sys,os,argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # so here won't have poll allocator info
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from img2feat_utils import getFiles, forward_graph, feat2layername


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("framelst",type=str,help="path to frame path list")
	parser.add_argument("featname",type=str,help="inception_resnet_v2 | inception_v4 | resnet_v2")
	parser.add_argument("modelpath",type=str,help="path to the checkpoint file")
	parser.add_argument("featpath",type=str,help="path to save the features. Each framepathname will be a npz file with item to be the framename.")
	parser.add_argument("--skip",action="store_true",help="whether to skip exsiting framepathname")
	parser.add_argument("--rename",default=False,action="store_true",help="given the framelst with 'framepath videoId', will name videoId.npz")

	# The following 3 only used for places365 image feature extraction, there is filename conflict across folder
	parser.add_argument("--savenpy",action="store_true",help="whether to store the feature into single npy under the path")
	parser.add_argument("--saveasgo",action="store_true",help="for one image path with a lot of images")

	parser.add_argument("--addpath2name",action="store_true",help="whether to add pathname to the filename, just for saving npy")
	parser.add_argument("--addpath2namelevel",type=int,default=1,help="add name from how many level up")

	parser.add_argument("--batchSize",type=int,default=10,help="batch_size")
	parser.add_argument("--l2norm",action="store_true",help="whether to do l2norm")
	parser.add_argument("--global_l2norm",action="store_true",help="apply l2norm **after** concat/exp show this is not good")

	parser.add_argument("--job",type=int,default=1,help="total job")
	parser.add_argument("--curJob",type=int,default=1,help="this script will execute job number")
	parser.add_argument("--no_logits",action="store_true",help="whether to not to add last layer (classification layer) logits as features")

	parser.add_argument("--show_endpoints",action="store_true",help="show model end points and then exit")
	parser.add_argument("--layer",type=str,default=None,help="this will over write the pre-defined layer")

	parser.add_argument("--format",default="jpg",help="image format")

	"""
		note for nasnet the model path is to model.ckpt
		chunwaileong@chunwaileong-ThundeRobot:~/cmu/semantics/test_feat$ python ~/cmu/semantics/script/img2feat.py video_frames.lst nasnet ~/cmu/semantics/models/nasnet/model.ckpt  test_nasnet_11262017 --l2norm
	"""


	return parser.parse_args()


if __name__ == "__main__":
	args = get_args()
	# construct the graph for inference
	try:
		input_place_holders, logits, end_points = forward_graph(args.featname,args.batchSize)
	except Exception as e:
		print "error loading graph:%s"%e
		sys.exit()



	if args.show_endpoints:
		for k in sorted(end_points.keys()):
			print k, end_points[k].get_shape()
		sys.exit()

	if not os.path.exists(args.featpath):
		os.makedirs(args.featpath)

	# the requested feature layername, will concat with last layer's logit
	image_size,num_class,layernames,_ = feat2layername[args.featname]
	# layernames is the predefined layer to extractg ['PreLogits'...]
	if args.layer is not None:
		layernames = [args.layer]

	runs = [end_points[layername] for layername in layernames]

	if not args.no_logits:
		runs.append(logits)

	framepaths = [line.strip() for line in open(args.framelst,"r").readlines()]

	count=0
	totalImgProcessed = 0
	tfconfig = tf.ConfigProto(allow_soft_placement=True)
	tfconfig.gpu_options.allow_growth = True
	with tf.Session(config=tfconfig) as sess:
		saver = tf.train.Saver()
		saver.restore(sess, args.modelpath)

		for framepath in tqdm(framepaths):
			count+=1
			#if(countPath % 200 == 0):
			#	print "%s folder processed,%s total" % (count,len(framepaths))
			if args.rename:
				framepath,videoId = framepath.split(" ")

			if((count % args.job) != (args.curJob-1)):
				continue

			framename = framepath.rstrip("/").split("/")[-1]

			if args.rename:
				framename = videoId

			if args.savenpy:
				if not args.rename:
					framename = "_".join(framepath.rstrip("/").split("/")[-args.addpath2namelevel:])

				target_feat_path = args.featpath
			else:
				# the target npz file we will generate
				target_feat_path = os.path.join(args.featpath,framename+".npz")

			# each frame feature will be in here
			feats = {}

			if args.skip and os.path.exists(target_feat_path):
				continue # skip existing frame

			images = getFiles(framepath,args.format)
			totalImgProcessed+=len(images)

			if len(images) == 0:
				tqdm.write("warning, %s has 0 imgs"%framename)
				continue

			# pack these images into batch size and feed
			countImg = 0
			if args.savenpy and args.saveasgo:
				if not os.path.exists(os.path.join(target_feat_path,framename)):
					os.makedirs(os.path.join(target_feat_path,framename))
				
			for i in xrange(0,len(images),args.batchSize):
				countImg+=args.batchSize
				#if countImg % 1000 == 0:
				#	print "%s jpg processed,%s total for folder %s/%s" % (countImg,len(images),count,len(framepaths))
				imgBatch = []
				for j in xrange(i,i+args.batchSize):
					if j < len(images):
						theImg = images[j]
					else:
						theImg = imgBatch[-1] # repeat to make a image batch full
					imgBatch.append(theImg)

				# feed the image batch

				#layer_values, logit_values = sess.run([end_points[layername],logits],feed_dict={input_place_holders:imgBatch})
				try:
					layer_values = sess.run(runs,feed_dict={input_place_holders:imgBatch})
				except Exception as e:
					print "warning, sess.run error, may due to image corrupt,img:%s,error message:%s"%(imgBatch,str(e)[:50])
					continue

				# save the feature
				for j in xrange(len(imgBatch)):
					# index for this feature
					imgname = os.path.splitext(os.path.basename(imgBatch[j]))[0]
					if feats.has_key(imgname): # so we don't waste computation for the last repeated images
						continue

					
					this_feat = []
					for k in xrange(len(layer_values)):
						if not args.global_l2norm and args.l2norm:
							l2norm_layer = np.linalg.norm(layer_values[k][j],2)
							layer_values[k][j]/=l2norm_layer
						this_feat.append(layer_values[k][j])

					this_feat = np.hstack(this_feat)
					if args.global_l2norm and args.l2norm:
						l2norm_layer = np.linalg.norm(this_feat,2)
						this_feat/=l2norm_layer

					feats[imgname] = this_feat

				if args.saveasgo and args.savenpy:
					for imgname in feats:
						save_to = os.path.join(target_feat_path,framename,imgname+".npy")
						np.save(save_to,feats[imgname])
					del feats
					feats ={}

			if not args.saveasgo:
				try:
					if args.savenpy:
						for imgname in feats:
							save_to = os.path.join(target_feat_path,framename+"_"+imgname+".npy")
							np.save(save_to,feats[imgname])
					else:
						np.savez_compressed(target_feat_path,**feats)
				except Exception as e:
					# might be filename too long
					tqdm.write("warning, %s saving error,:%s"%(framename,str(e)[:40]))
					pass
				del feats
	print "total image processed:%s, avg %s image per path"%(totalImgProcessed,totalImgProcessed/float(len(framepaths)))












