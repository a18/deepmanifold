#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import io
import glob
import os
import os.path
import sys
skimage_io_import_bug_workaround=sys.argv
try:
  sys.argv=sys.argv[:1]
  import skimage.io
finally:
  sys.argv=skimage_io_import_bug_workaround
  del skimage_io_import_bug_workaround

import yaml
import random
import urllib2
import urlparse
import time
import subprocess
import pipes

import caffe.io
import matchmmd
import dmt
from gen_deepart import read_lfw_attributes,attr_pairs
from gen_deepart import minibatch
from gen_deepart import setup_classifier
from gen_deepart import deepart_reconstruct
from gen_deepart import ratelimit
from imageutils import montage

if __name__=='__main__':
  print('PPID/PID {} {}'.format(os.getppid(),os.getpid()))
  print(os.uname()[1])
  sys.stdout.flush()
  args=sys.argv[1:]
  assert len(args)==1
  
  # read experiment config
  with open(args[0],'r') as f:
    config=yaml.safe_load(f.read())
  print(yaml.dump(config))
  prefix=os.path.splitext(args[0])[0]

  # silly yaml, it doesn't parse 1e8 as a float
  weights=[float(x) for x in config['weights']]
  rbf_var=float(config['rbf_var'])

  with open('dataset/lfw.txt','r') as f:
    filelist=['images/'+x.strip() for x in f.readlines()]

  if config['colorize']:
    indices=list(range(len(filelist)))
    random.seed(123)
    random.shuffle(indices)
    source_indices=indices[:config['source_k']]
    target_indices=indices[config['source_k']:config['source_k']+config['target_k']]
  else:
    _,lfwattrname,lfwattr=read_lfw_attributes()
    attr=lfwattrname.index(config['attribute'])
    if config['reversed']:
      source_indices,target_indices=attr_pairs(lfwattr,attr,config['source_k'],config['target_k'])
    else:
      target_indices,source_indices=attr_pairs(lfwattr,attr,config['target_k'],config['source_k'])

  # test_set is random images not in the P or Q
  indices=sorted(list(set(range(len(filelist)))-set(source_indices)-set(target_indices)))
  random.seed(123)
  random.shuffle(indices)
  test_indices=indices[:config['test_k']]
  if 0 not in test_indices: test_indices[0]=0 # The Aaron Eckhart Project

  if config['colorize']:
    ipath1=[filelist[i].replace('lfw','lfw_gray') for i in source_indices]
    ipath2=[filelist[i] for i in target_indices]
    ipath3=[]
    ipath4=[filelist[i].replace('lfw','lfw_gray') for i in test_indices]
  else:
    ipath1=[filelist[i] for i in source_indices]
    ipath2=[filelist[i] for i in target_indices]
    ipath3=[]
    ipath4=[filelist[i] for i in test_indices]

  N=len(ipath1)
  M=len(ipath2)
  L=len(ipath3)
  ipath=ipath1+ipath2+ipath3+ipath4

  XF,F2,root_dir,result=dmt.run(ipath,N,M,L,config['model'],config['image_dims'],config['device_id'],weights,rbf_var,prefix,config['num_iter'],False,config['zscore'])
  with open('{}/config.yaml'.format(root_dir),'w') as f:
    f.write(yaml.dump(config))
