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
  basename=urlparse.urlparse(url).path.replace('/','_')
  if basename in blacklist:
    raise IOError
  subdir=os.path.splitext(basename)[0][-1]
  opath='images/imagenet/{}/{}'.format(subdir,basename)
  if not os.path.exists(os.path.split(opath)[0]):
    os.makedirs(os.path.split(opath)[0])
  if not os.path.exists(opath):
    try:
      time.sleep(0.2) # don't hammer the server
      r=requests.get(url)
      with open(opath,'wb') as f:
        f.write(r.content)
      I=skimage.io.imread(opath)
      return opath
    except IOError:
      pass
    try:
      time.sleep(1) # second try, give them some time so their server can wake up
      r=requests.get(url)
      with open(opath,'wb') as f:
        f.write(io.BytesIO(r.content))
      I=skimage.io.imread(opath)
    except IOError:
      raise
  else:
    try:
      I=skimage.io.imread(opath)
    except IOError:
      raise
  return opath

def extract(S,featext,model,image_dims,device_id,blob_names):
  rlprint=ratelimit(interval=60)(print)
  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)
    
  # minibatch processing
  M=10
  work_units,work_done,work_t0=len(S),0,time.time()
  for i,inputs in enumerate(minibatch(S,M)):
    F=net.extract_features(inputs,blob_names,auto_reshape=True)
    for j,ipath in enumerate(inputs):
      opath=os.path.splitext(ipath)[0]+featext
      with open(opath,'wb') as f:
        numpy.savez(f,conv3_1=F['conv3_1'][j],conv4_1=F['conv4_1'][j],conv5_1=F['conv5_1'][j])
    work_done=work_done+len(inputs)
    rlprint('{}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))

def form_F(ipath,featext,blob_names):
  # to recover blob k: F[F_slice[k]].reshape(*F_shape[k])
  F_shape={}
  F_slice={}
  F=[]
  for i,x in enumerate(ipath):
    data=numpy.load(os.path.splitext(x)[0]+featext)
    if i==0:
      for k in blob_names:
        F_shape[k]=data[k].shape
    F.append(numpy.concatenate([data[k].ravel() for k in blob_names]))
  F=numpy.array(F)
  index=0
  for k in blob_names:
    F_slice[k]=slice(index,index+numpy.prod(F_shape[k]))
    index=index+numpy.prod(F_shape[k])
  return F,F_slice,F_shape

if __name__=='__main__':
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
  P=ipath[:N]
  Q=ipath[N:N+M]
  X=ipath[N+M:]

  print(ipath)
  print('{} source, {} target, {} test'.format(N,M,len(X)))

  blob_names=['conv3_1','conv4_1','conv5_1']

  # extract features
  featext='.{}.{}x{}.npz'.format(config['model'],*config['image_dims'])
  S=list(set(x for x in ipath if not os.path.exists(os.path.splitext(x)[0]+featext)))
  if len(S)>0:
    print('S',S)
    extract(S,featext,config['model'],config['image_dims'],config['device_id'],blob_names)

  weights=[float(x) for x in config['weights']] # silly yaml, cannot parse 8e-8

  # Form F (first N rows are P, next M rows are Q, last row is x)
  F,F_slice,F_shape=form_F(ipath,featext,blob_names)
  print('F',F.shape)
  print(F_slice)
  print(F_shape)
  XF=F[N+M:]

  # Solve for multiple points on the manifold (move away from P toward Q)
  allF2=[]
  for x in XF:
    F[N+M]=x
    XPR,R=matchmmd.manifold_traversal(F[:N+M+1],N,M,weights,rbf_var=1e10,checkgrad=False,checkrbf=True)
    print('R',R.shape,R.dtype,R.sum(axis=1))
    allF2.append(XPR.dot(F[:N+M+1]))
  F2=numpy.asarray(allF2,dtype=numpy.float32)
  F2=F2.reshape(F2.shape[0]*F2.shape[1],-1)
  print('F2',F2.shape,F2.dtype,F2.min(),F2.max())

  print('XF',XF.shape,XF.dtype)
  dataset_F=numpy.concatenate([XF,F2],axis=0)
  data_indices=range(len(X),len(X)+len(F2))
  test_indices=list(numpy.repeat(range(len(X)),len(weights)))
  print('dataset_F',dataset_F.shape,dataset_F.dtype)
  print('data_indices',data_indices)
  print('test_indices',test_indices)
 
  deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=1000,test_indices=test_indices,data_indices=data_indices,image_dims=config['image_dims'],hybrid_names=['conv1_1','conv2_1'],hybrid_weights=[0.02,0.02],dataset=X,dataset_F=dataset_F,dataset_slice=F_slice,dataset_shape=F_shape,desc=prefix)
