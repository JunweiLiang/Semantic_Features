# coding=utf-8 
# uniform frame extraction from a videolst
# note that timeTicks are all integer second

import sys,os,argparse,commands
from tqdm import tqdm
# need ffmpeg and ffprbe

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("videolst")
	parser.add_argument("framepath")
	parser.add_argument("--num_per_video",type=int,default=30)
	parser.add_argument("--job",type=int,default=1)
	parser.add_argument("--curJob",type=int,default=1)
	parser.add_argument("--skiperror",action="store_true",help="whether to skip videos that we can't get duration, default get all first frame if no duration")
	parser.add_argument("--fixnum",action="store_true",help="whether to get num_per_video no matter what")
	parser.add_argument("--keyframe",action="store_true",help="whether to extract keyframe for longer video")
	parser.add_argument("--longsec",type=float,default=60.0,help="if extract keyframe, video longer than seconds")
	return parser.parse_args()

def getDuration(video):
	output = commands.getoutput("ffprobe -i '%s' -show_entries format=duration -v quiet -of csv='p=0'"%video).strip() # will output empty if errors occurs
	duration = None
	try:
		duration = float(output)
	except Exception as e:
		tqdm.write("warning, %s duration error"%(os.path.basename(video)))
		duration = None
	return duration

def getUniformTicks(duration,num_frames,fixnum=False):
	if duration is None: # can't get duration
		if fixnum:
			ticks = [0.0 for i in xrange(num_frames)] # then get all first frame
		else:
			ticks = [0.0]
	else:
		# interger second since ffmpeg input accepts no
		step = int(duration/float(num_frames))
		if step == 0:
			step = 1
		ticks,time = [],0.0
		for i in xrange(num_frames):
			ticks.append(time)
			time+=step
			if time > duration:
				if fixnum:
					time=duration
				else:
					break
	return ticks

def sec2time(secs): # no .second
	m,s = divmod(secs,60)
	#print m,s
	h,m = divmod(m,60)
	s = int(s)
	return "%02d:%02d:%02d"%(h,m,s)

"""
		ffmpeg -ss 00:00:05 -i input.mp4
	       -ss 00:01:05 -i input.mp4
	       -ss 00:03:05 -i input.mp4
	       -ss 00:40:05 -i input.mp4 
	       -map 0:v -frames:v 1 out001.jpg
	       -map 1:v -frames:v 1 out002.jpg
	       -map 2:v -frames:v 1 out003.jpg
	       -map 3:v -frames:v 1 out004.jpg
"""


if __name__ == "__main__":
	args = get_args()

	# check keyframe lib
	if args.keyframe:
		# path to the tool kits, including VideoSegHisto and all its libs
		libpath = "/home/chunwaileong/cmu/semantics/keyframe_tools/lib"

		if not os.path.exists(libpath):
			print "lib path not exists! check it"
			sys.exit()

	if not os.path.exists(args.framepath):
		os.makedirs(args.framepath)

	videos = [line.strip() for line in open(args.videolst,"r").readlines()]

	keyframeVideoCount = 0

	errorDuration=[]

	count=0
	for video in tqdm(videos):
		count+=1
		if((count % args.job) != (args.curJob-1)):
			continue
		if not os.path.exists(video):
			tqdm.write("warning, %s not exists"%video)
			continue

		

		duration = getDuration(video) # in seconds

		if args.skiperror and (duration is None):
			errorDuration.append(video)
			continue

		videoname = os.path.splitext(os.path.basename(video))[0]
		newframepath = os.path.join(args.framepath,videoname)
		if not os.path.exists(newframepath):
			os.makedirs(newframepath)


		if args.keyframe and (duration > args.longsec):
			keyframeVideoCount+=1
			cmd = "LD_LIBRARY_PATH=%s && %s/VideoSegHisto '%s' '%s' 0x11 30 2 0 0.33 1"%(libpath,libpath,video,newframepath)
			output = commands.getoutput(cmd)

		else:

			# all time in seconds needed to extract frame
			timeTicks = [sec2time(tick) for tick in getUniformTicks(duration,args.num_per_video,args.fixnum)]

			# make the command string
			cmd = " ".join(["-ss %s -i '%s'"%(tick,video) for tick in timeTicks])
			cmd+=" "+" ".join(["-map %d:v -frames:v 1 '%s/rgb_%06d.jpg'"%(i,newframepath,i+1) for i in xrange(len(timeTicks))])

			#print cmd
			output = commands.getoutput("ffmpeg -y %s"%cmd)
			#print output
			#sys.exit()
	if args.skiperror:
		print "total %s videos, %s duration cannot get"%(len(videos),len(errorDuration))
	if args.keyframe:
		print "total %s videos, %s extract keyframes"%(len(videos),keyframeVideoCount)


