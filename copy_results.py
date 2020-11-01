# coding=utf-8
"""Copy top k resulting frames based on the query results"""
import os
import argparse
import sys
if sys.version_info > (3, 0):
  import subprocess as commands
else:
  import commands

parser = argparse.ArgumentParser()
parser.add_argument("result_file")
parser.add_argument("frame_path")
parser.add_argument("--topk", type=int, default=50)
parser.add_argument("target_path")
parser.add_argument("--video_path", default=None)
parser.add_argument("--target_video_path", default=None)
parser.add_argument("--video_topk", type=int, default=10)

if __name__ == "__main__":
  args = parser.parse_args()

  if not os.path.exists(args.target_path):
    os.makedirs(args.target_path)

  if args.target_video_path is not None:
    if not os.path.exists(args.target_video_path):
      os.makedirs(args.target_video_path)

  for i, line in enumerate(open(args.result_file).readlines()[:args.topk]):
    videoname, simi, frame_name = line.strip().split(",")

    frame_file = os.path.join(args.frame_path, videoname, "%s.jpg" % frame_name)

    commands.getoutput("cp %s %s/%d_%s.jpg" % (
        frame_file, args.target_path, i+1, videoname))

    if args.video_path is not None and args.target_video_path is not None:
      if i+1 > args.video_topk:
        continue
      commands.getoutput("cp %s/%s.mp4 %s/%d_%s.mp4" % (
          args.video_path, videoname, args.target_video_path, i+1, videoname))


