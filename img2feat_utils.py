# coding=utf-8
# all image2feat model

import sys,os

# path to git clone https://github.com/tensorflow/models/  stuff
#slimpath = '/home/chunwaileong/cmu/semantics/tf_code/slim/'
#slimpath = '/home/junweil/cmu/semantics/tf_code/slim/' # vid-gpu3
slimpath = '/home/junweil/semantics/tf_code/slim/'
if not os.path.exists(slimpath):
	print "slim path not exists!"
	sys.exit()
sys.path.append(slimpath)

import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from nets.nasnet.nasnet import nasnet_large_arg_scope
from nets.nasnet.nasnet import build_nasnet_large as nasnet
# not using this
#from nets.inception_v4 import inception_v4_arg_scope, inception_v4
#from nets.resnet_v2 import resnet_v2_152,resnet_arg_scope


feat2layername = {
	#"resnet_v2_152":"pool5",  # https://arxiv.org/abs/1603.05027
	#"inception_v4":"PreLogitsFlatten", # https://arxiv.org/pdf/1602.07261.pdf
	"inception_resnet_v2":(299,1001,["PreLogitsFlatten"],"inception_resnet_v2_arg_scope"),
	"nasnet":(331,1001,["global_pool"],"nasnet_large_arg_scope"), # 4032 
	"resnet_v2_152":(299,1001,["pool5"],"resnet_arg_scope"),
	#"inception_v4":"Predictions",
	#"inception_resnet_v2":"Predictions"
}


# modified resnet_v2 to have end_points pool5 and Predictions

def forward_graph(featname,batchSize):
	
	assert featname in feat2layername.keys(),feat2layername.keys()
	image_size,num_class,end_points,arg_scope_name = feat2layername[featname]

	with tf.device("/cpu:0"):
		# add prepro stuff, input to tensorflow is image file name
		jpeg_data_names = tf.placeholder(tf.string,shape=(batchSize,))
		# for image with the same size
		decode_jpeg = tf.map_fn(lambda imgstr:
			tf.image.convert_image_dtype(
				tf.image.decode_jpeg(tf.read_file(imgstr), channels=3),
			dtype=tf.float32),
		jpeg_data_names,dtype=tf.float32,back_prop=False,infer_shape=True,parallel_iterations=10)

		image = tf.image.resize_bilinear(decode_jpeg, [image_size,image_size], align_corners=False) # resnet, inception v3
		"""
		preprocessing_fn_map = {
	      'cifarnet': cifarnet_preprocessing,
	      'inception': inception_preprocessing,
	      'inception_v4': inception_preprocessing,
	      'inception_resnet_v2': inception_preprocessing,
	      'lenet': lenet_preprocessing,
	      'mobilenet_v1': inception_preprocessing,
	      'nasnet_mobile': inception_preprocessing,
	      'nasnet_large': inception_preprocessing,
	      'resnet_v1_50': vgg_preprocessing,
	      'resnet_v1_101': vgg_preprocessing,
	      'resnet_v1_152': vgg_preprocessing,
	  }
		"""

		scaled_input_tensor = tf.subtract(image, 0.5)
		scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)

	# the computational graph
	#arg_scope_name = "%s_arg_scope"%featname
	arg_scope = globals()[arg_scope_name]()
	feat_net = globals()[featname]

	with slim.arg_scope(arg_scope):
		logits, end_points = feat_net(scaled_input_tensor, is_training=False,num_classes=num_class)

	return jpeg_data_names, logits, end_points


def getFiles(pathtop,type_):
	pathtop = pathtop.rstrip("/")
	pathList = os.listdir(pathtop)
	files = []
	for path in pathList:
		path = pathtop + '/' + path
			
		#是否指定type
		if(type_ == ""):
			files.append(path)
		else:
			if(os.path.basename(path).endswith("."+type_)):
				files.append(path)
	return files

