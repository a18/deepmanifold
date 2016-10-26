#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
np=numpy
import time
import os
import os.path
import collections
import sys
skimage_io_import_bug_workaround=sys.argv
try:
  sys.argv=sys.argv[:1]
  import skimage.io
finally:
  sys.argv=skimage_io_import_bug_workaround
  del skimage_io_import_bug_workaround

import matchmmd
import imageutils
from gen_deepart import minibatch
from gen_deepart import setup_classifier
from gen_deepart import deepart_reconstruct
from gen_deepart import ratelimit

def chunking_dot(big_matrix, small_matrix, chunk_size=10000):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((small_matrix.shape[0],big_matrix.shape[1]))
    for i in range(0, R.shape[1], chunk_size):
        end = i + chunk_size
        R[:,i:end] = np.dot(small_matrix, big_matrix[:,i:end])
    return R

def extract(S,featext,model,image_dims,device_id,blob_names):
  '''S: a list of image paths
featext: image features will be stored in files with this suffix
model: name of CNN model
image_dims: 2-tuple of height, width
device_id: GPU device id, zero indexed
blob_names: names of blobs to extract, must be in forward order
'''
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
        numpy.savez(f,**dict([(k,F[k][j]) for k in blob_names]))
    work_done=work_done+len(inputs)
    rlprint('extract {}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))

def form_F(ipath,featext,blob_names):
  '''ipath: a list of paths to images
featext: image features will be read from files with this suffix
blob_names: names of blobs to read, in order

Returns F, F_slice and F_shape. Each row of F is a concatenation of the
flattened blobs. To recover blob k: F[i,F_slice[k]].reshape(*F_shape[k])
'''
  #rlprint=ratelimit(interval=60)(print)
  F_shape={}
  F_slice={}
  K=len(ipath)
  #work_units,work_done,work_t0=len(ipath),0,time.time()
  for i,x in enumerate(ipath):
    data=numpy.load(os.path.splitext(x)[0]+featext)
    if i==0:
      for k in blob_names:
        F_shape[k]=data[k].shape
      D=sum([numpy.prod(F_shape[k]) for k in blob_names])
      F=numpy.zeros((K,D),dtype=data[k].dtype)
    F[i]=numpy.concatenate([data[k].ravel() for k in blob_names])
    #work_done=work_done+1
    #rlprint('dmt {}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))
  index=0
  for k in blob_names:
    F_slice[k]=slice(index,index+numpy.prod(F_shape[k]))
    index=index+numpy.prod(F_shape[k])
  return F,F_slice,F_shape

def reconstruct_traversal(ipath,device_id):
  '''Given the reconstruction values from a previous call to
dmt.run() with traversal_only=True, this function performs the
reconstruction.

Returns XF, F2, root_dir and result. XF is the feature matrix of the
original images. F2 is the feature matrix of the transformed images
(len(weights) transformed images per original image). root_dir is the
name of the results directory. result is the transformed images.
'''
  data=numpy.load(ipath)

  XF=data['XF']
  F2=data['F2']
  weights=list(data['weights'])
  hybrid=bool(data['hybrid'])
  blob_names=list(data['blob_names'])
  hybrid_blob_names=list(data['hybrid_blob_names'])
  prefix=str(data['prefix'])
  max_iter=str(data['max_iter'])
  test_indices=list(data['test_indices'])
  data_indices=list(data['data_indices'])
  image_dims=list(data['image_dims'])
  X=list(data['X'])
  dataset_F=data['dataset_F']
  F_slice=data['F_slice'].reshape(-1)[0]
  F_shape=data['F_shape'].reshape(-1)[0]

  if hybrid:
    root_dir,result=deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=max_iter,test_indices=test_indices,data_indices=data_indices,image_dims=image_dims,hybrid_names=hybrid_blob_names,hybrid_weights=[0.02]*len(hybrid_blob_names),dataset=X,dataset_F=dataset_F,dataset_slice=F_slice,dataset_shape=F_shape,desc=prefix,device_id=device_id)
  else:
    root_dir,result=deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=max_iter,test_indices=test_indices,data_indices=data_indices,image_dims=image_dims,dataset=X,dataset_F=dataset_F,dataset_slice=F_slice,dataset_shape=F_shape,desc=prefix,device_id=device_id)

  result=numpy.array(result)
  print('result',result.shape,result.dtype)
  result=result.reshape(len(XF),len(weights),*result.shape[1:])
  originals=numpy.array([skimage.io.imread('{}/{}-original.png'.format(root_dir,os.path.splitext(os.path.split(x)[1])[0]))/255.0 for x in X])
  originals=originals.reshape(1,*originals.shape)
  m=result.transpose((1,0)+tuple(range(2,result.ndim))) # stack vertically
  M=imageutils.montage(originals)
  skimage.io.imsave('{}/montage_originals.png'.format(root_dir),M)
  M=imageutils.montage(m)
  skimage.io.imsave('{}/montage_results.png'.format(root_dir),M)
  M=imageutils.montage(numpy.concatenate([originals,m]))
  skimage.io.imsave('{}/montage_all.png'.format(root_dir),M)

  return XF,F2,root_dir,result

def run(ipath,N,M,L,model,image_dims,device_id,weights,rbf_var,prefix,max_iter,hybrid,zscore,maxnumlinesearch=25,traversal_only=False,blob_names=['conv3_1','conv4_1','conv5_1'],hybrid_blob_names=['conv1_1','conv2_1']):
  '''This function will take a list of paths to images and run deep
manifold traversal. First, features are extracted if needed. Next,
the manifold traversal of each image is optimized. Lastly, the images
are reconstructed and visualized. Visual results appear in a directory
named result_TIMESTAMP_PREFIX.

ipath: a list of image paths
N: The first N images are P, the source
M: The next M images are Q, the target
L: The next L images are T, the data
len(X): The remaining images are X, the images to transform.
model: Name of CNN model
image_dims: 2-tuple of height, width
device_id: GPU device id, zero indexed
weights: A list of lambda weights for the budget-of-change regularizer
rbf_var: Variance for the rbf kernel
prefix: Used in the name of the results directory
max_iter: Number of iterations for the reconstruction
hybrid: True if you want to the use the layer regularizer
zscore: True if you want to zscore F
maxnumlinesearch: Number of conjugate gradient line searches
traversal_only: Does not perform reconstruction
blob_names: list of blobs for the manifold traversal
hybrid_blob_names: list of blobs for the layer regularizer

If traversal_only is False:

  Returns XF, F2, root_dir and result. XF is the feature matrix of the
  original images. F2 is the feature matrix of the transformed images
  (len(weights) transformed images per original image). root_dir is the
  name of the results directory. result is the transformed images.

If traversal_only is True:

  Returns XF, F2, traversal. traversal is an NPZ file containing the
  values needed for reconstruction.
'''
  rlprint=ratelimit(interval=60)(print)
  t0=time.time()

  P=ipath[:N]
  Q=ipath[N:N+M]
  T=ipath[N+M:N+M+L]
  X=ipath[N+M+L:]

  print('{} source, {} target, {} data, {} test'.format(N,M,L,len(X)))
  assert N>0 and M>0 and L>=0 and len(X)>0

  # extract features
  featext='.{}.{}x{}.npz'.format(model,*image_dims)
  S=list(set(x for x in ipath if not os.path.exists(os.path.splitext(x)[0]+featext)))
  if len(S)>0:
    extract(S,featext,model,image_dims,device_id,blob_names)

  # Form F (first N rows are P, next M rows are Q, next L are T, last row is x)
  print('Forming F ...')
  F,F_slice,F_shape=form_F(ipath,featext,blob_names)
  if zscore:
    print('Z scoring ...')
    loc,sigma=matchmmd.zscore_F(F)
  print('F',F.shape,F.dtype)
  print(F_slice)
  print(F_shape)
  XF=F[N+M+L:]
  print('Computing FFT1 ...')
  FFT1=F[:N+M+L+1].dot(F[:N+M+L+1].T)
  #print('FFT1 fro',numpy.linalg.norm(FFT1,'fro'))

  # Solve for multiple points on the manifold (move away from P toward Q)
  allF2=[]
  work_units,work_done,work_t0=len(XF),0,time.time()
  for x in XF:
    F[N+M+L]=x
    nv=F[:N+M+L].dot(x)
    FFT1[:-1,-1]=nv
    FFT1[-1,:-1]=nv
    FFT1[-1,-1]=x.dot(x)
    XPR,R=matchmmd.manifold_traversal2(FFT1,N,M,L,weights,rbf_var=rbf_var,checkgrad=False,checkrbf=True,verbose=False,maxnumlinesearch=maxnumlinesearch)
    print('R',R.shape,R.dtype,R.sum(axis=1))
    print('Reprojecting ...')
    if zscore:
      allF2.append(chunking_dot(F[:N+M+L+1],XPR)*sigma+loc)
    else:
      allF2.append(chunking_dot(F[:N+M+L+1],XPR))
    work_done=work_done+1
    rlprint('dmt {}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))
  if zscore:
    XF=XF*sigma+loc
  F2=numpy.asarray(allF2,dtype=numpy.float32)
  print('F2',F2.shape,F2.dtype,F2.min(),F2.max())
  # save deep manifold traversal result
  #with open('{}_dmt.npz'.format(prefix),'wb') as f: numpy.savez(f,PF=F[:N],QF=F[N:N+M],TF=F[N+M:N+M+L],XF=XF,F2=F2,XPR=XPR,R=R,F_shape=F_shape,weights=weights,rbf_var=rbf_var)

  F2=F2.reshape(F2.shape[0]*F2.shape[1],-1)
  dataset_F=numpy.concatenate([XF,F2],axis=0)
  data_indices=range(len(X),len(X)+len(F2))
  test_indices=list(numpy.repeat(range(len(X)),len(weights)))

  if traversal_only:
    opath='traversal_{}_{}.npz'.format(int(round(t0)),prefix)
    with open(opath,'wb') as f: numpy.savez(f,XF=XF,F2=F2,weights=weights,hybrid=hybrid,hybrid_blob_names=hybrid_blob_names,blob_names=blob_names,prefix=prefix,max_iter=max_iter,test_indices=test_indices,data_indices=data_indices,image_dims=image_dims,X=X,dataset_F=dataset_F,F_slice=F_slice,F_shape=F_shape)
    print('Wrote',opath)
    print('{} minutes.'.format((time.time()-t0)/60.0))
    return XF,F2,opath
 
  if hybrid:
    root_dir,result=deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=max_iter,test_indices=test_indices,data_indices=data_indices,image_dims=image_dims,hybrid_names=hybrid_blob_names,hybrid_weights=[0.02]*len(hybrid_blob_names),dataset=X,dataset_F=dataset_F,dataset_slice=F_slice,dataset_shape=F_shape,desc=prefix,device_id=device_id)
  else:
    root_dir,result=deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=prefix,max_iter=max_iter,test_indices=test_indices,data_indices=data_indices,image_dims=image_dims,dataset=X,dataset_F=dataset_F,dataset_slice=F_slice,dataset_shape=F_shape,desc=prefix,device_id=device_id)

  result=numpy.array(result)
  print('result',result.shape,result.dtype)
  result=result.reshape(len(XF),len(weights),*result.shape[1:])
  originals=numpy.array([skimage.io.imread('{}/{}-original.png'.format(root_dir,os.path.splitext(os.path.split(x)[1])[0]))/255.0 for x in X])
  originals=originals.reshape(1,*originals.shape)
  m=result.transpose((1,0)+tuple(range(2,result.ndim))) # stack vertically
  M=imageutils.montage(originals)
  skimage.io.imsave('{}/montage_originals.png'.format(root_dir),M)
  M=imageutils.montage(m)
  skimage.io.imsave('{}/montage_results.png'.format(root_dir),M)
  M=imageutils.montage(numpy.concatenate([originals,m]))
  skimage.io.imsave('{}/montage_all.png'.format(root_dir),M)

  print('{} minutes.'.format((time.time()-t0)/60.0))
  return XF,F2,root_dir,result
