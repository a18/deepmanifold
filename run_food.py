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
import requests
import time
import subprocess
import pipes

import caffe.io
import matchmmd
import dmt
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
  
  test_set=sorted(glob.glob('images/food-101/images/{}/*.jpg'.format(config['source'])))
  random.seed(123)
  random.shuffle(test_set)
  ipath4=test_set[:10]
  ipath1=[x for x in sorted(glob.glob('images/food-101/images/{}/*.jpg'.format(config['source']))) if x not in ipath4]
  ipath2=[x for x in sorted(glob.glob('images/food-101/images/{}/*.jpg'.format(config['target']))) if x not in ipath4]
  ipath3=[]
  for y in config['data']:
    ipath3.extend([x for x in sorted(glob.glob('images/food-101/images/{}/*.jpg'.format(y))) if x not in ipath4])
  random.seed(123)
  random.shuffle(ipath1)
  random.seed(123)
  random.shuffle(ipath2)

  N=len(ipath1)
  M=len(ipath2)
  L=len(ipath3)
  ipath=ipath1+ipath2+ipath3+ipath4

  XF,F2,root_dir,result=dmt.run(ipath,N,M,L,config['model'],config['image_dims'],config['device_id'],weights,rbf_var,prefix,config['num_iter'],False)
  with open('{}/config.yaml'.format(root_dir),'w') as f:
    f.write(yaml.dump(config))
