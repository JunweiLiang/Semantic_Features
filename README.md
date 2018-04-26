# CMU Video-Level Semantic Features

This repository contains the inferencing code and models for this paper:

    Po-Yao Huang, Ye Yuan, Zhenzhong Lan, Lu Jiang, and Alexander G. Hauptmann
    "Video Representation Learning and Latent Concept Mining for Large-scale Multi-label Video Classification"
    in arXiv preprint arXiv:1707.01408 (2017).


## Dependencies
+ Python 2.7; TensorFlow >= 1.4.0; tqdm and nltk (for preprocessing)
+ Pre-trained [models](https://aladdin1.inf.cs.cmu.edu/shares/semantic_features/models_04262018.tgz) (1.1GB). Extract the models into path models/

## What the code does
Given a list of videos, output the semantic features for each video.


## Inferencing
1. Extract frames from each video
```
$ python extractFrames_uniform.py videos.lst frames_path --num_per_video 30
```

2. Extract frame-level CNN features

First change the slimpath in img2feat_utils.py

```
$ python img2feat.py frames_path.lst inception_resnet_v2 models/inception_resnet_v2.ckpt frame_feature_path --l2norm --batchSize 30
```

3. Average pooling into video-level CNN features
```
$ python aversgeFeats.py frame_feature_path.lst video_feature_path --l2norm 
```

4. Extract semantic features

First change the modelpath in semantics_features.py

```
$ python semantics_features.py video_feature_path.lst semantic_feat --save_seperate
```