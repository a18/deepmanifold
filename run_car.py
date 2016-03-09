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
  
  cars=sorted([os.path.split(x)[1].split('_')[1] for x in glob.glob('images/car24r/*mesh024*.png')])
  print(cars)
  random.seed(123)
  random.shuffle(cars)
  test_car=cars[:5]
  for index in range(len(test_car)):
    ipath1=[]
    ipath2=[]
    for x in cars:
      if x==test_car[index]: continue
      ipath1.extend(['images/car24r/car_{}_mesh{}.png'.format(x,z) for z in config['source']])
      ipath2.extend(['images/car24r/car_{}_mesh{}.png'.format(x,z) for z in config['target']])
    ipath3=['images/car24r/car_{}_mesh{}.png'.format(test_car[index],config['source'][len(config['source'])//2])]
    N=len(ipath1)
    M=len(ipath2)
    ipath=ipath1+ipath2+ipath3
    print(ipath)
    XF,F2,root_dir,result=dmt.run(ipath,N,M,0,config['model'],config['image_dims'],config['device_id'],weights,rbf_var,prefix+'_'+test_car[index],config['num_iter'],False,False)
    A=skimage.io.imread(root_dir+'/car_{}_mesh{}-original.png'.format(test_car[index],config['source'][len(config['source'])//2]))/255.0
    B=skimage.io.imread('images/car24r/car_{}_mesh{}.png'.format(test_car[index],config['target'][len(config['target'])//2]))/255.0
    B=caffe.io.resize_image(B,A.shape)
    C=[skimage.io.imread(x)/255.0 for x in sorted(glob.glob(root_dir+'/car_{}_mesh{}-0*.png'.format(test_car[index],config['source'][len(config['source'])//2])))]
    M=numpy.array([A]+[B]+C)
    M=M.reshape(-1,1,*M.shape[1:])
    M=montage(M)
    skimage.io.imsave(root_dir+'/result_{}.png'.format(test_car[index]),M)
    with open('{}/config.yaml'.format(root_dir),'w') as f:
      f.write(yaml.dump(config))
