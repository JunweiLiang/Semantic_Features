# coding=utf-8
"""Given query npz, compute cosine/l2 similarity for the search set."""

import sys, os, argparse, operator

from glob import glob
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("search_path")
parser.add_argument("result")
parser.add_argument("--dist_metric", default="cosine")

if __name__ == "__main__":
  args = parser.parse_args()

  query = np.load(args.query)
  query_frame = sorted(query.keys())[len(query.keys()) // 2]

  query_vector = query[query_frame]

  # 2537
  #print(query_vector.shape)
  print("Using query frame %s from %s.." % (
      query_frame, os.path.basename(args.query)))
  dists = []  # videoname -> highest similarity , frame_name

  search_files = glob(os.path.join(args.search_path, "*.npz"))

  for search_file in tqdm(search_files):
    videoname = os.path.splitext(os.path.basename(search_file))[0]

    search_data = np.load(search_file)

    this_dists = []  #simi, frame_name

    for frame_name in search_data:
      search_vector = search_data[frame_name]

      if args.dist_metric == "l2":
        # l2 similarity
        simi = - np.linalg.norm(query_vector - search_vector)  # lower closer -> higher closer
      elif args.dist_metric == "cosine":
        # cosine similarity
        simi = np.dot(query_vector, search_vector)/(
            np.linalg.norm(query_vector)*np.linalg.norm(search_vector))
      else:
        raise Exception("Not implemented metric: %s" % args.dist_metric)

      this_dists.append((simi, frame_name))

    this_dists.sort(key=operator.itemgetter(0), reverse=True)

    dists.append((videoname, this_dists[0][0], this_dists[0][1]))

  dists.sort(key=operator.itemgetter(1), reverse=True)

  with open(args.result, "w") as f:
    for videoname, simi, frame_name in dists:
      f.writelines("%s,%.5f,%s\n" % (videoname, simi, frame_name))
  print("top 10:")
  print("\t%s" % dists[:10])


