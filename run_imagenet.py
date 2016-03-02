#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
import io
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

import matchmmd
import dmt
from gen_deepart import minibatch
from gen_deepart import setup_classifier
from gen_deepart import deepart_reconstruct
from gen_deepart import ratelimit

def fetch_imagenet_urls(S):
  result=[]
  for synset in S:
    url='http://image-net.org/api/text/imagenet.synset.geturls?wnid={}'.format(synset)
    f=urllib2.urlopen(url)
    try:
      result.extend(x.strip() for x in f.readlines())
    finally:
      f.close()
  return result

blacklist=[
 '_3636_3361762172_9f581f7ec0.jpg'
]

def download_if_needed(url):
  basename=urlparse.urlparse(url).path.replace('/','_').replace('~','_')
  if basename in blacklist:
    raise IOError
  try:
    subdir=os.path.splitext(basename)[0][-1]
  except:
    subdir='_'
  opath='images/imagenet/{}/{}'.format(subdir,basename)
  if not os.path.exists(os.path.split(opath)[0]):
    os.makedirs(os.path.split(opath)[0])
  if not os.path.exists(opath):
    print('Downloading {}'.format(url))
    try:
      time.sleep(0.2) # don't hammer the server
      r=requests.get(url,timeout=30)
      with open(opath,'wb') as f:
        f.write(r.content)
      I=skimage.io.imread(opath)
      if I.ndim!=1 and I.ndim!=3: raise IOError
      return opath
    except IOError:
      pass
    try:
      time.sleep(1) # second try, give them some time so their server can wake up
      r=requests.get(url,timeout=30)
      with open(opath,'wb') as f:
        f.write(r.content)
      I=skimage.io.imread(opath)
      if I.ndim!=1 and I.ndim!=3: raise IOError
      return opath
    except IOError:
      # write a zero-length file to skip future download attempts
      with open(opath,'wb') as f:
        pass
      raise
  else:
    try:
      I=skimage.io.imread(opath)
      if I.ndim!=1 and I.ndim!=3: raise IOError
    except IOError:
      raise
  return opath

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
  source_k=config['source_k']
  target_k=config['target_k']
  test_k=config['test_k']
  
  # read synset lists
  Purl=fetch_imagenet_urls(config['source'])
  Qurl=fetch_imagenet_urls(config['target'])
  Xurl=fetch_imagenet_urls(config['test'])
  if len(Purl)<source_k:
    print('WARNING: not enough source images! ({} available, {} requested)'.format(len(Purl),source_k))
  if len(Qurl)<target_k:
    print('WARNING: not enough target images! ({} available, {} requested)'.format(len(Qurl),target_k))
  if len(Xurl)<test_k:
    print('WARNING: not enough test images! ({} available, {} requested)'.format(len(Xurl),test_k))
  sys.stdout.flush()
  
  # randomly select k
  random.seed(123)
  random.shuffle(Purl)
  random.seed(123)
  random.shuffle(Qurl)
  random.seed(123)
  random.shuffle(Xurl)
  Purl=Purl[:source_k]
  Qurl=Qurl[:target_k]
  Xurl=Xurl[:test_k]
  # force X to not be in the "training" set
  Purl=[x for x in Purl if x not in Xurl]
  Qurl=[x for x in Qurl if x not in Xurl]

  # download images (if needed)
  ipath=[]
  N=0
  for url in Purl:
    try:
      ipath.append(download_if_needed(url))
      N=N+1
    except IOError:
      pass
  M=0
  for url in Qurl:
    try:
      ipath.append(download_if_needed(url))
      M=M+1
    except IOError:
      pass
  for url in Xurl:
    try:
      ipath.append(download_if_needed(url))
    except IOError:
      pass

  # silly yaml, it doesn't parse 1e8 as a float
  weights=[float(x) for x in config['weights']]
  rbf_var=float(config['rbf_var'])

  XF,F2,root_dir,result=dmt.run(ipath,N,M,config['model'],config['image_dims'],config['device_id'],weights,rbf_var,prefix,config['num_iter'],False)
  with open('{}/config.yaml'.format(root_dir),'w') as f:
    f.write(yaml.dump(config))
