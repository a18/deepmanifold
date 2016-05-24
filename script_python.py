
## Imports

import numpy
np=numpy
import inspect
import resource
import ast
import yaml
import sys
skimage_io_import_bug_workaround=sys.argv
try:
  sys.argv=sys.argv[:1]
  import skimage.io
finally:
  sys.argv=skimage_io_import_bug_workaround
  del skimage_io_import_bug_workaround
import skimage.restoration
import sklearn.decomposition
import pickle
import time
import glob
import os
import os.path
import subprocess
import pipes
import h5py
import itertools
import scipy
import seaborn
import pandas
import matplotlib.pyplot
import totalvariation
import threadparallel

import measure
import deepart
import matchmmd

## Read PCA matrices

#data=np.load('pca.npz') # matlab
data=np.load('data_pca.npz') # python
U=data['U']
T=data['T']
mu=data['mu']
shape=data['shape'].item()
del data
print 'U',U.shape,U.dtype,U.min(),U.max()
print 'T',T.shape,T.dtype,T.min(),T.max()
print 'mu',mu.shape,mu.dtype,mu.min(),mu.max()

## Read attributes

from gen_deepart import read_lfw_attributes
from gen_deepart import attr_pairs

_,_,lfwattr=read_lfw_attributes()
# 8 is Youth
# 10 is Senior
target_indices,source_indices=attr_pairs(lfwattr,10,2000,2000)
print 'source',source_indices[:5]
print 'target',target_indices[:5]

weights=[5e-6]
test_indices=[0,1]
print 'feat_img[:5]',T[0,:5]
#P=T[source_indices].astype(np.float64)
#Q=T[target_indices].astype(np.float64)
P=T[source_indices]
Q=T[target_indices]
print 'P',P.shape,P.dtype
print 'Q',Q.shape,Q.dtype

## Match distributions

allF=[]
for i in test_indices:
  #reload(matchmmd)
  t0=time.time()
  #x_0,x,r=matchmmd.match_distribution(T[i].astype(np.float64),P,Q,weights)
  x_0,x,r=matchmmd.match_distribution(T[i],P,Q,weights)
  print('Finished in {} minutes.'.format((time.time()-t0)/60.0))
  print('x_0',x_0.shape,x_0.dtype,x_0.min(),x_0.max())
  print('x',x.shape,x.dtype,x.min(),x.max())
  print('r',r.shape,r.dtype,r.min(),r.max())
  F=x.dot(U)+mu
  print('F',F.shape,F.dtype,F.min(),F.max())
  print(F[:,:10])
  allF.append(F)
  # debug stuff
  x_0=x_0.dot(U)+mu
  print('X',x_0.shape,x_0.dtype,x_0.min(),x_0.max())
  R=r.dot(U)
  print('R',R.shape,R.dtype,R.min(),R.max())

prefix='out'
blob_names=['conv3_1','conv4_1','conv5_1']
F=np.asarray(allF,dtype=np.float32)
F=F.reshape(F.shape[0]*F.shape[1],-1)
print('F',F.shape,F.dtype,F.min(),F.max())
index=0
for k in blob_names:
  size=numpy.prod(shape[k][1:])
  h5f=h5py.File('{}_{}.h5'.format(prefix,k),'w')
  h5f.create_dataset('DS',data=F[:,index:index+size].reshape(*shape[k]))
  h5f.close()
  index=index+size

## Reconstruct

from gen_deepart import deepart_reconstruct

deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=3000,test_indices=test_indices,data_indices=None,image_dims=(125,125),hybrid_names=['conv1_1','conv2_1'],hybrid_weights=[0.02,0.02],dataset='lfw',desc='testpy')

