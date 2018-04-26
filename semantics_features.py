# coding=utf-8
# edit the model list and path, and index2class file path, given list of feature to predict, 
# return a json, with each feature to each model, each class's prob


import sys,os,argparse,json,operator,math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # so here won't have poll allocator info
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("featlst")
	parser.add_argument("output")
	parser.add_argument("--showTop3",action="store_true",help="for debug, show each feat each model top3 predictions")
	parser.add_argument("--save_seperate",action="store_true",help="use this when feature list is large, saving to each individual files")
	return parser.parse_args()

# put all model, index2class file into one path
modelpath = "/home/junweil/cmu/semantics/datasets/" # vid-gpu3
#modelpath = "/mnt/sdb/junweil/semantics/" # vid-gpu1
# edit this to omit models

models = {
	"kinetics":(os.path.join(modelpath,"kinetics","MoRe16x1_packed"),os.path.join(modelpath,"kinetics","index2class.lst")),
	"fcvid":(os.path.join(modelpath,"fcvid","MoRe16x1_packed"),os.path.join(modelpath,"fcvid","index2class.lst")),
	"ucf101":(os.path.join(modelpath,"ucf101","MoRe16x1_packed"),os.path.join(modelpath,"ucf101","index2class.lst")),
	"sin346":(os.path.join(modelpath,"sin346","MoRe16x1_packed"),os.path.join(modelpath,"sin346","index2class.lst")),
	"places365":(os.path.join(modelpath,"places365","MoRe16x1_packed"),os.path.join(modelpath,"places365","index2class.lst")),
	"sports487":(os.path.join(modelpath,"sports487","MoRe16x1_packed"),os.path.join(modelpath,"sports487","index2class.lst")),
	"yfcc609":(os.path.join(modelpath,"yfcc609","MoRe16x1_packed"),os.path.join(modelpath,"yfcc609","index2class.lst")),
	"moments":(os.path.join(modelpath,"moments","MoRe32x1_packed"),os.path.join(modelpath,"moments","index2class.lst")),
	"imagenet_shuffle":(os.path.join(modelpath,"imagenet_shuffle","MoRe16x1_packed"),os.path.join(modelpath,"imagenet_shuffle","index2class.lst")),
}

featDim = 2537

def chunk(lst,n):
	for i in xrange(0,len(lst),n):
		yield lst[i:i+n]

def predict(featfilelst,models,args):
	batch_size = len(featfilelst)
	input_ = np.zeros((batch_size,featDim),dtype="float32")
	predictions = {}

	for i,featfile in enumerate(featfilelst):
		input_[i,:] = np.load(featfile)
	"""
	models[modelname] = {
			"Model":Model(modelpath),
			"index2class":loadIndex2class(index2classpath), # {index:classname} # zero based
			"name":modelname
		}
	"""
	for modelname in models:
		model = models[modelname]["Model"]
		predictions[modelname] = model.run(input_)

	# construct filename -> modelname -> predictions
	output = {}
	for i,featfile in enumerate(featfilelst):
		# assume no filename conflicts
		filename = os.path.splitext(os.path.basename(featfile))[0]
		output[filename] = {}
		for modelname in models:
			output[filename][modelname] = mapPredict(predictions[modelname][i],models[modelname]['index2class'],args)
			if args.showTop3:
				print filename,modelname,output[filename][modelname][:3]
	return output

def mapPredict(predictions,index2class,args):
	out = []
	for i in xrange(len(predictions)):
		out.append({"classname":index2class[i],"score":float(predictions[i])})
		#out[index2class[i]] = float(predictions[i])
	# not sort to save time
	if args.showTop3:
		out.sort(key=operator.itemgetter("score"),reverse=True)
	return out

def loadIndex2class(index2classpath):
	index2class = {}
	for line in open(index2classpath,"r").readlines():
		index,classname = line.strip().split(" ",1)
		index2class[int(index) - 1] = classname # to zero based
	return index2class

# for each model, use different graph and session to run
#  If you are using more than one graph (created with tf.Graph() in the same process, you will have to use different sessions for each graph, but each graph can be used in multiple sessions. In this case, it is often clearer to pass the graph to be launched explicitly to the session constructor.
# https://www.tensorflow.org/versions/r0.12/api_docs/python/client/session_management
class Model:
	def __init__(self,modelpath):
		self.graph = tf.Graph()
		tfconfig = tf.ConfigProto(allow_soft_placement=True)
		tfconfig.gpu_options.allow_growth = True 
		self.sess = tf.Session(config=tfconfig,graph=self.graph)

		with self.graph.as_default() as graph:
			# so model path should has a  :"final_model_packed/checkpoint"
			if not os.path.exists(modelpath):
				raise Exception("model not exists:%s"%modelpath)
			modelpath = tf.train.get_checkpoint_state(modelpath).model_checkpoint_path
			
			saver = tf.train.import_meta_graph(modelpath+".meta")

			saver.restore(self.sess,modelpath)

			# save path is one.pb file
			"""
			with tf.gfile.GFile(modelpath,"rb") as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())

			print [n.name for n in graph_def.node]
			tf.import_graph_def(
				graph_def,
				return_elements=None
			)
			"""

			# input place holders
			self.input = tf.get_collection("input")[0]
			self.is_train = tf.get_collection("is_train")[0] # TODO: remove this

			self.output = tf.get_collection("output")[0]
			#print [n.name for n in tf.get_default_graph().as_graph_def().node]

			# TODO: make this more elegant
			model_note_var = [v for v in tf.global_variables() if v.name == "model_note:0"][0]
			self.sess.run(tf.variables_initializer([model_note_var]))
			self.modelnote = self.sess.run(model_note_var)

			print "loaded %s, note:%s"%(modelpath,self.modelnote)

	def run(self,input_):
		return self.sess.run(self.output,feed_dict={self.input:input_,self.is_train:False})

	def done(self):
		self.sess.close()



def loadModels(models):

	for modelname in models:
		modelpath,index2classpath = models[modelname]
		models[modelname] = {
			"Model":Model(modelpath),
			"index2class":loadIndex2class(index2classpath), # {index:classname} # zero based
			"name":modelname
		}
	return models


if __name__ == "__main__":
	args = get_args()

	featlst = [line.strip() for line in open(args.featlst,"r").readlines()]

	output = {}
	if args.save_seperate:
		if not os.path.exists(args.output):
			os.makedirs(args.output)

	batch_size = 1024

	models = loadModels(models) # each model has its own tf.Graph() and tf.Session()

	for feats in tqdm(chunk(featlst,batch_size),total=int(math.ceil(len(featlst)/float(batch_size)))):	
		
		predicts = predict(feats,models,args)
		# uses base filename, so assume no filename confict
		if args.save_seperate:
			for filename in predicts:
				json.dump(predicts[filename],open(os.path.join(args.output,filename+".json"),"w"))
		else:
			output.update(predicts) # filename -> sorted(models) -> sorted(predictions[classname:prob])

	if not args.save_seperate:
		json.dump(output,open(args.output,"w"))

	for modelname in models:
		models[modelname]["Model"].done() # close all session