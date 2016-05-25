#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import os
import os.path
import sys
import yaml
import random
import time

import dmt

def parse_args(args):
  if len(args)<1 or '--help' in args:
    print('''Usage:

dmt.py --source [SOURCE1 ...] --target [TARGET1 ...] --data [DATA1 ...] --image [IMAGE1 ...]

Specify four sets of paths to images. Source and target are the images
which define the source and target submanifolds. Data is an optional
set of images which define additional basis vectors. Image is the set
of images to transform.
''')
    sys.exit(1)
  active=None
  N,M,L,X=[],[],[],[]
  for x in args:
    if x=='--source' or x=='-N':
      active=N
    elif x=='--target' or x=='-M':
      active=M
    elif x=='--data' or x=='-L':
      active=L
    elif x=='--image' or x=='-X':
      active=X
    elif active is None:
      raise ValueError('Unexpected argument (specify --source, --target, --data or --image): {}'.format(x))
    else:
      if not os.path.exists(x):
        raise IOError(x)
      active.append(x)
  return N,M,L,X

if __name__=='__main__':
  args=sys.argv[1:]
  N,M,L,X=parse_args(args)
  
  ipath=N+M+L+X
  model='vgg'
  image_dims=[250,250]
  device_id=0

  weights=[1e-7]

  # HOWTO set weights: Use grid search. The result you want (smallest
  # change that is perceivable) will probably be between one and zero.

  rbf_var=7.7e5

  # HOWTO set rbf_var: Run DMT. Look at the KP and KQ values (5 values,
  # mean and variance are reported). Set rbf_var so that mean is reasonable
  # (e.g., 0.5). 7.7e5 is a guess for small images (250x250). 2.5e7 is a
  # guess for large images (900x600)

  prefix='dmt'
  num_iter=2000
  blob_names=['conv3_1','conv4_1','conv5_1']

  XF,F2,root_dir,result=dmt.run(ipath,len(N),len(M),len(L),model,image_dims,device_id,weights,rbf_var,prefix,num_iter,False,True,blob_names=blob_names,hybrid_blob_names=[])
  print('Results are in {}'.format(root_dir))

