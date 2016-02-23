
## Imports

import numpy
np=numpy
import sys
import pickle
import time
import os
import os.path
import subprocess
import pipes
import h5py
import itertools

import matchmmd
from gen_deepart import read_lfw_attributes,attr_pairs
from gen_deepart import deepart_reconstruct

## Read P, Q, X, weights and layers

colorization=False
attr=10
source_k=2000
target_k=2000
test_indices=sorted([6005, 3659, 8499, 12217, 9982, 4322, 10449, 10969, 4245, 7028])
test_indices=[0,176]
_,_,lfwattr=read_lfw_attributes()
if attr>=0:
  target_indices,source_indices=attr_pairs(lfwattr,attr,target_k,source_k)
else:
  source_indices,target_indices=attr_pairs(lfwattr,-attr,source_k,target_k)

if colorization:
  source_indices=list(numpy.random.choice(range(len(lfwattr)),source_k,replace=False))
  target_indices=list(numpy.random.choice(range(len(lfwattr)),target_k,replace=False))

P=sorted(source_indices) # list of indices (source distribution)
Q=sorted(target_indices) # list of indices (target distribution)
X=test_indices   # list of indices (test images)
weights=[8e-8,4e-8,2e-8,1e-8]

## Form F (first N rows are P, next M rows are Q, last row is x)

prefix='data'
blob_names=['conv3_1','conv4_1','conv5_1']
h5f={}
shape={}
for k in blob_names:
  assert os.path.exists('{}_{}.h5'.format(prefix,k))
  h5f[k]=h5py.File('{}_{}.h5'.format(prefix,k),'r')
  print('h5f',k,h5f[k]['DS'].shape,h5f[k]['DS'].dtype)
  N=h5f[k]['DS'].shape[0]
  shape[k]=h5f[k]['DS'].shape[1:]

PF=numpy.concatenate([h5f[k]['DS'][P].reshape(len(P),-1) for k in blob_names],axis=1)
QF=numpy.concatenate([h5f[k]['DS'][Q].reshape(len(Q),-1) for k in blob_names],axis=1)
XF=numpy.concatenate([h5f[k]['DS'][X].reshape(len(X),-1) for k in blob_names],axis=1)
for k in blob_names:
  h5f[k].close()

if colorization:
  prefix='gray'
  blob_names=['conv3_1','conv4_1','conv5_1']
  h5f={}
  shape={}
  for k in blob_names:
    assert os.path.exists('{}_{}.h5'.format(prefix,k))
    h5f[k]=h5py.File('{}_{}.h5'.format(prefix,k),'r')
    print('h5f',k,h5f[k]['DS'].shape,h5f[k]['DS'].dtype)
    N=h5f[k]['DS'].shape[0]
    shape[k]=h5f[k]['DS'].shape[1:]
  PF=numpy.concatenate([h5f[k]['DS'][P].reshape(len(P),-1) for k in blob_names],axis=1)
  XF=numpy.concatenate([h5f[k]['DS'][X].reshape(len(X),-1) for k in blob_names],axis=1)
  for k in blob_names:
    h5f[k].close()

F=numpy.concatenate([PF,QF,XF[:1]],axis=0)
print 'F',F.shape,F.dtype
del h5f,PF,QF

## Solve for multiple points on the manifold (move away from P toward Q)

reload(matchmmd)
allF2=[]
allF3=[]
for x in XF:
  F[-1]=x
  XPR,R=matchmmd.manifold_traversal(F,len(P),len(Q),weights,rbf_var=1e10,checkgrad=False,checkrbf=True)
  print('R',R.shape,R.dtype)
  if True:
    XPR2=numpy.copy(XPR)
    XPR2[:,:len(P)]=XPR[:,:len(P)].mean(axis=1,keepdims=True)
    XPR2[:,len(P):len(P)+len(Q)]=XPR[:,len(P):len(P)+len(Q)].mean(axis=1,keepdims=True)
    print('weight variance',XPR[:,:len(P)].var(axis=1),XPR[:,len(P):len(P)+len(Q)].var(axis=1))
    print('linear weights',XPR2[:,0],XPR2[:,len(P)],XPR[:,-1])
    allF3.append(XPR2.dot(F))
  allF2.append(XPR.dot(F))

prefix='out'
F2=np.asarray(allF2,dtype=np.float32)
F2=F2.reshape(F2.shape[0]*F2.shape[1],-1)
print('F2',F2.shape,F2.dtype,F2.min(),F2.max())
index=0
for k in blob_names:
  size=numpy.prod(shape[k])
  h5f=h5py.File('{}_{}.h5'.format(prefix,k),'w')
  h5f.create_dataset('DS',data=F2[:,index:index+size].reshape(-1,*shape[k]))
  h5f.close()
  index=index+size

prefix='out2'
F3=np.asarray(allF3,dtype=np.float32)
F3=F3.reshape(F3.shape[0]*F3.shape[1],-1)
print('F3',F3.shape,F3.dtype,F3.min(),F3.max())
index=0
for k in blob_names:
  size=numpy.prod(shape[k])
  h5f=h5py.File('{}_{}.h5'.format(prefix,k),'w')
  h5f.create_dataset('DS',data=F3[:,index:index+size].reshape(-1,*shape[k]))
  h5f.close()
  index=index+size

## Reconstruct input from points in latent space

prefix='out'
test_indices2=list(numpy.repeat(test_indices,len(weights)))
deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=3000,test_indices=test_indices2,data_indices=None,image_dims=(125,125),hybrid_names=['conv1_1','conv2_1'],hybrid_weights=[0.02,0.02],dataset='lfw',desc='testpy_XPR')

prefix='out2'
test_indices2=list(numpy.repeat(test_indices,len(weights)))
deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=3000,test_indices=test_indices2,data_indices=None,image_dims=(125,125),hybrid_names=['conv1_1','conv2_1'],hybrid_weights=[0.02,0.02],dataset='lfw',desc='testpy_XPR2')

