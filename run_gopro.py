#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import io
import glob
import os
import os.path
import subprocess
import pipes
import shutil
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
from gen_deepart import read_lfw_attributes,attr_pairs,attr_positive,attr_negative,attr_read_named
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

  with open('gopro_daynight/gopro_daynight.txt') as f:
    L = f.readlines()
  L = ['gopro_daynight/' + l.strip() for l in L]

  source_indices = range(*config['source_range'])
  target_indices = range(*config['target_range'])
  # pick random source images for test
  random.seed(123)
  random.shuffle(source_indices)
  test_indices=source_indices[:config['test_k']]
  # ensure test does not appear in source or target
  source_indices=list(set(source_indices)-set(test_indices))
  target_indices=list(set(target_indices)-set(test_indices))

  everything_else_ipath = []
  source_ipath = [L[i] for i in source_indices]
  target_ipath = [L[i] for i in target_indices]
  test_ipath = [L[i] for i in test_indices]

  ipath = source_ipath + target_ipath + everything_else_ipath + test_ipath
  N = len(source_ipath)
  M = len(target_ipath)
  L = len(everything_else_ipath)

  if config['traversal_only']:
    # run this on high memory CPU
    # (you must have already extracted features)
    XF,F2,traversal=dmt.run(ipath,N,M,L,config['model'],config['image_dims'],config['device_id'],weights,rbf_var,prefix,config['num_iter'],config['hybrid'],config['zscore'],traversal_only=True)
    with open('{}_config.yaml'.format(os.path.splitext(traversal)[0]),'w') as f:
      f.write(yaml.dump(config))
  elif config['reconstruct_only']:
    # run this on high memory GPU
    # (you must have already done traversal)
    traversal=config['reconstruct_only']
    if not os.path.exists(traversal):
      # fetch traversal from high memory CPU machine
      subprocess.check_call(['rsync','-avzP','mrmaster:/isis2b/git/deepmanifold/{}'.format(traversal),'.'])
      subprocess.check_call(['rsync','-avzP','mrmaster:/isis2b/git/deepmanifold/{}_config.yaml'.format(os.path.splitext(traversal)[0]),'.'])
    XF,F2,root_dir,result=dmt.reconstruct_traversal(traversal,config['device_id'])
    shutil.copy('{}_config.yaml'.format(os.path.splitext(traversal)[0]),'{}/config.yaml'.format(root_dir))
  else:
    # run this on high memory CPU+GPU
    # (will also extract features, if needed)
    XF,F2,root_dir,result=dmt.run(ipath,N,M,L,config['model'],config['image_dims'],config['device_id'],weights,rbf_var,prefix,config['num_iter'],config['hybrid'],config['zscore'])
    with open('{}/config.yaml'.format(root_dir),'w') as f:
      f.write(yaml.dump(config))
