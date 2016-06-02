#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy
np=numpy
import collections
import threading
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

from fet_extractor import load_fet_extractor
from deepart import gen_target_data, optimize_img
from test_deepart import test_all_gradients
import measure
import deepart
import matchmmd
import imageutils
import threadparallel
import models

from gen_deepart import setup_classifier

def invert_model(filelist,targetlist,model,image_dims=None,max_iter=3000,hybrid_names=[],hybrid_weights=[],tv_lambda=0.001,tv_beta=2,desc='identity',device_id=0):
  # filelist is a list of images
  # targetlist is a list of targets
  # each target is a (tname,blob_names,blob_weights)
  # tname is a descriptive string
  # blob_names is a list of blobs, must be in forward-to-back order
  # blob_weights is a list of scalar weights, one for each blob
  #
  # Example: targetlist=[('c5',['conv5_1'],[1])]
  # This example will try to reconstruct an image by finding the image which
  # matches the conv5_1 features of that image.
  #
  # The script will test reconstruction for each target for each image.
  t0=time.time()

  # init result dir
  root_dir='results_{}'.format(int(round(t0))) if desc=='' else 'results_{}_{}'.format(int(round(t0)),desc)
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  def print(*args):
    with open('{}/log.txt'.format(root_dir),'a') as f:
      f.write(' '.join(str(x) for x in args)+'\n')
    sys.stdout.write(' '.join(str(x) for x in args)+'\n')

  print('filelist',filelist)
  print('targetlist',targetlist)
  print('model',model)
  print('image_dims',image_dims)
  print('max_iter',max_iter)
  print('tv_lambda',tv_lambda)
  print('tv_beta',tv_beta)
  print('desc',desc)
  print('device_id',device_id)

  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)

  all_results=[]

  for tname,blob_names,blob_weights in targetlist:

    all_psnr=[]
    all_ssim=[]
  
    for ipath1 in filelist:
  
      np.random.seed(123)
  
      basename=os.path.splitext(os.path.split(ipath1)[1])[0]
      root_dir2='{}/{}/{}'.format(root_dir,model,tname)
      if not os.path.exists(root_dir2):
        os.makedirs(root_dir2)
  
      all_target_blob_names=list(hybrid_names)+list(blob_names)
      targets=[]
      target_data_list=[]
      F=net.extract_features([ipath1],all_target_blob_names,auto_reshape=True)
      for k,v in zip(hybrid_names,hybrid_weights):
        if len(targets)>0 and targets[-1][3]==v:
          targets[-1][1].append(k)
          target_data_list[-1][k]=F[k]
        else:
          targets.append((None,[k],False,v))
          target_data_list.append({k: F[k]})
        print('hybrid',k,v,F[k].shape,F[k].dtype)
      for k,v in zip(blob_names,blob_weights):
        if len(targets)>0 and targets[-1][3]==v:
          targets[-1][1].append(k)
          target_data_list[-1][k]=F[k]
        else:
          targets.append((None,[k],False,v))
          target_data_list.append({k: F[k]})
        print('blob',k,v,F[k].shape,F[k].dtype)
  
      # load ground truth
      A=caffe.io.load_image(ipath1) # ground truth
      B=net.preprocess_inputs([A],auto_reshape=True)
      C=net.transformer.deprocess(net.inputs[0],B)
      D=caffe.io.resize_image(C,A.shape) # best possible result (only preprocess / deprocess)
      print('input',A.shape,A.dtype,A.min(),A.max())
      print('pre',B.shape,B.dtype,B.min(),B.max())
      print('de',C.shape,C.dtype,C.min(),C.max())
      print('re',D.shape,D.dtype,D.min(),D.max())
  
      # optimize
      # Set initial value and reshape net
      init_img=np.random.normal(loc=0.5,scale=0.1,size=A.shape)
      deepart.set_data(net,init_img)
      #x0=np.ravel(init_img).astype(np.float64)
      x0=net.get_input_blob().ravel().astype(np.float64)
      bounds=zip(np.full_like(x0,-128),np.full_like(x0,162))
      solver_type='L-BFGS-B'
      solver_param={'maxiter': max_iter, 'iprint': -1}
      opt_res=scipy.optimize.minimize(deepart.objective_func,x0,args=(net,all_target_blob_names,targets,target_data_list,tv_lambda,tv_beta),bounds=bounds,method=solver_type,jac=True,options=solver_param)
      data=np.reshape(opt_res.x,net.get_input_blob().shape)[0]
      deproc_img=net.transformer.deprocess(net.inputs[0],data)
      Dhat=caffe.io.resize_image(np.clip(deproc_img,0,1),A.shape)
      all_results.append(Dhat)

      # evaluate
      print('{} {} best psnr = {:.4}, ssim = {:.4}'.format(tname,basename,measure.measure_PSNR(A,D,1).mean(),measure.measure_SSIM(A,D,1).mean()))
      psnr=measure.measure_PSNR(A,Dhat,1).mean()
      ssim=measure.measure_SSIM(A,Dhat,1).mean()
      print('{} {} actual psnr = {:.4}, ssim = {:.4}'.format(tname,basename,psnr,ssim))
      skimage.io.imsave('{}/{}_original.png'.format(root_dir2,basename),A)
      skimage.io.imsave('{}/{}_best.png'.format(root_dir2,basename),D)
      skimage.io.imsave('{}/{}_actual.png'.format(root_dir2,basename),Dhat)
      caption='psnr {:.4}, ssim {:.4}'.format(psnr,ssim)
      subprocess.check_call('convert {root_dir2}/{basename}_original.png {root_dir2}/{basename}_actual.png -size {w}x -font Arial-Italic -pointsize 12 caption:{caption} -append {root_dir2}/eval_{basename}.png'.format(root_dir2=pipes.quote(root_dir2),basename=pipes.quote(basename),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)
      all_psnr.append(psnr)
      all_ssim.append(ssim)
  
    print(tname,'psnr',psnr)
    print(tname,'ssim',ssim)
    mean_psnr=np.asarray(all_psnr).mean()
    mean_ssim=np.asarray(all_ssim).mean()
    with open('{}/autoencoder.txt'.format(root_dir),'a') as f:
      f.write('{},{},{},{}\n'.format(model,tname,mean_psnr,mean_ssim))

  t1=time.time()
  print('Finished in {} minutes.'.format((t1-t0)/60.0))
  return root_dir,np.asarray(all_results)

if __name__=='__main__':
  filelist=['images/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg']
  targetlist=[
    ('c3c4c5',['conv3_1','conv4_1','conv5_1'],[1,1,1]),
  ]
  model='vgg'
  root_dir,result=invert_model(filelist,targetlist,model,image_dims=[100,100],max_iter=3000,tv_lambda=0.001,tv_beta=2,desc='invert',device_id=0)
  original=np.asarray([skimage.io.imread(x)/255.0 for x in filelist*len(targetlist)])
  m=np.asarray([original,result])
  skimage.io.imsave('{}/montage.png'.format(root_dir),imageutils.montage(m))

