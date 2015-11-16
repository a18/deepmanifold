#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import numpy as np
import sys
import skimage.io
import time
import glob
import os
import os.path
import subprocess
import pipes
import h5py
import itertools
import scipy

from fet_extractor import load_fet_extractor
from deepart import gen_target_data, optimize_img
from test_deepart import test_all_gradients
import measure
import deepart

def ratelimit(n=0,interval=0.0,timefn=time.time,blocking=False,blockingfn=time.sleep):
  def d(f):
    count=[n,interval,timefn,blockingfn]
    if interval>0.0: count[1]=count[2]()-interval
    def c(*args,**kwds):
      if n>0: count[0]=count[0]+1
      t=count[1]
      if interval>0.0: t=count[2]()
      if blocking and interval>t-count[1]:
        count[3](interval-(t-count[1]))
        t=interval+count[1]
      if count[0]>=n and t>=interval+count[1]:
        count[0]=0
        count[1]=t
        f(*args,**kwds)
    return c
  return d

def minibatch(x,n):
  it=iter(x)
  while True:
    x=list(itertools.islice(it,n))
    if len(x)<1: break
    yield x

def setup_classifier(model='vgg',image_dims=(224,224),device_id=1):
    #deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_deepart.prototxt'
    #weights_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers.caffemodel'
    #image_dims = (1014//2, 1280//2)
    #mean = (104, 117, 123)

    if model=='vgg':
        deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
        weights_relpath = 'models/VGG_CNN_19/vgg_normalised.caffemodel'
        mean = (103.939, 116.779, 123.68)
    elif model=='vggface':
        deployfile_relpath = 'models/vgg_face_caffe/VGG_FACE_deploy_conv.prototxt'
        weights_relpath = 'models/vgg_face_caffe/VGG_FACE.caffemodel'
        mean = (93.5940, 104.7624, 129.1863)
    else:
        raise ValueError('Unknown CNN model:',model)
    input_scale = 1.0

    caffe, net = load_fet_extractor(
        deployfile_relpath, weights_relpath, image_dims, mean, device_id,
        input_scale
    )

    return caffe, net, image_dims


def run_deepart(ipath1='images/starry_night.jpg',ipath2='images/tuebingen.jpg',max_iter=2000):
    np.random.seed(123)

    root_dir = 'results_{}'.format(int(round(time.time())))
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    display = 100
    # list of targets defined by tuples of
    # (
    #     image path,
    #     target blob names (these activations will be included in the loss function),
    #     whether we use style (gram) or content loss,
    #     weighting factor
    # )
#    targets = [
#        (ipath1, ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], True, 1),
#        (ipath2, ['conv4_2'], False, 1),
#    ]
#    targets = [
#        (ipath1, ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1'], False, 1),
#    ]
    targets = [
        (ipath1, ['conv5_1'], False, 1),
    ]
    # These have to be in the same order as in the network!
    all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']

    caffe, net, image_dims = setup_classifier()

    A=caffe.io.load_image(args[0])
    B=net.preprocess_inputs([A],auto_reshape=True)
    C=net.transformer.deprocess(net.inputs[0],B)
    D=caffe.io.resize_image(C,A.shape)
    print('input',A.shape,A.dtype,A.min(),A.max())
    print('pre',B.shape,B.dtype,B.min(),B.max())
    print('de',C.shape,C.dtype,C.min(),C.max())
    print('re',D.shape,D.dtype,D.min(),D.max())
    print('psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,D,1).mean(),measure.measure_SSIM(A,D,1).mean()))

    # Generate activations for input images
    target_data_list = gen_target_data(root_dir, caffe, net, targets)

    # Generate white noise image
    init_img = np.random.normal(loc=0.5, scale=0.1, size=image_dims + (3,))

    solver_type = 'L-BFGS-B'
    solver_param = {}

    #test_all_gradients(init_img, net, all_target_blob_names, targets, target_data_list)

    Chat=optimize_img(
        init_img, solver_type, solver_param, max_iter, display, root_dir, net,
        all_target_blob_names, targets, target_data_list
    )
    Dhat=caffe.io.resize_image(Chat,A.shape)
    print('best psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,D,1).mean(),measure.measure_SSIM(A,D,1).mean()))
    print('actual psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean()))
    skimage.io.imsave('{}/eval_original.png'.format(root_dir),A)
    skimage.io.imsave('{}/eval_best.png'.format(root_dir),D)
    skimage.io.imsave('{}/eval_actual.png'.format(root_dir),Dhat)
    caption='psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean())
    subprocess.check_call('convert {root_dir}/eval_best.png {root_dir}/eval_actual.png -size {w}x -font Arial-Italic -pointsize 14 caption:{caption} -append {root_dir}/eval.png'.format(root_dir=pipes.quote(root_dir),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)

def deepart2(ipath1,ipath2,init_img=None,display=100,root_dir='results',max_iter=2000):
    targets = [
        (ipath1, ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'], True, 100),
        (ipath2, ['conv4_2'], False, 1),
    ]
    # These have to be in the same order as in the network!
    all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
    caffe, net, image_dims = setup_classifier()

    # Generate activations for input images
    target_data_list = gen_target_data(root_dir, caffe, net, targets)

    # Generate white noise image
    if init_img == None:
        init_img = np.random.normal(loc=0.5, scale=0.1, size=image_dims + (3,))

    solver_type = 'L-BFGS-B'
    solver_param = {}

    optimize_img(
        init_img, solver_type, solver_param, max_iter, display, root_dir, net,
        all_target_blob_names, targets, target_data_list
    )

def deepart_identity(max_iter=1000):
  # Experimenting with making deepart produce the identify function
  t0=time.time()
  display = 100
  # These have to be in the same order as in the network!
  all_target_blob_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

  ipathset = [
    'images/celebrity_tr/59_Mr._T_0004.jpg',
    'images/celebrity_tr/56_James_Read_0014.jpg',
    'images/celebrity_tr/57_Stephen_Lang_0008.jpg',
    'images/celebrity_tr/50_Denzel_Washington_0013.jpg',
    'images/celebrity_tr/57_Michael_Bowen_0012.jpg',
    'images/celebrity_tr/61_Annie_Golden_0006.jpg',
    'images/celebrity_tr/55_Jamey_Sheridan_0006.jpg',
    'images/celebrity_tr/54_Hulk_Hogan_0007.jpg',
    'images/celebrity_tr/56_Gus_Van_Sant_0007.jpg',
    'images/celebrity_tr/57_Chuck_Lorre_0001.jpg',
  ]

  ipathset = [
    'images/lfw/Charles_Moose/Charles_Moose_0013.jpg',
    'images/lfw/Eddy_Merckx/Eddy_Merckx_0002.jpg',
    'images/lfw/Taufik_Hidayat/Taufik_Hidayat_0001.jpg',
    'images/lfw/John_Travolta/John_Travolta_0005.jpg',
    'images/lfw/Hitomi_Soga/Hitomi_Soga_0001.jpg',
    'images/lfw/Lino_Oviedo/Lino_Oviedo_0001.jpg',
    'images/lfw/Daniel_Montgomery/Daniel_Montgomery_0001.jpg',
    'images/lfw/Paul_Burrell/Paul_Burrell_0004.jpg',
    'images/lfw/David_Beckham/David_Beckham_0031.jpg',
    'images/lfw/Caroline_Dhavernas/Caroline_Dhavernas_0001.jpg',
  ]

  targetset=[
    ('c5',[ ['', ['conv5_1'], False, 1], ]),
    ('c4',[ ['', ['conv4_1'], False, 1], ]),
    ('c3',[ ['', ['conv3_1'], False, 1], ]),
    ('c45',[ ['', ['conv4_1', 'conv5_1'], False, 1], ]),
    ('c345',[ ['', ['conv3_1', 'conv4_1', 'conv5_1'], False, 1], ]),
    #('c55',[ ['', ['conv5_1', 'conv5_2'], False, 1], ]),
    #('c4455',[ ['', ['conv4_1', 'conv4_2', 'conv5_1', 'conv5_2'], False, 1], ]),
    #('c555',[ ['', ['conv5_1', 'conv5_2', 'conv5_3'], False, 1], ]),
    #('c444555',[ ['', ['conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'], False, 1], ]),
  ]

  for model in ['vggface','vgg']:

    caffe, net, image_dims = setup_classifier(model=model)

    for tname,targets in targetset:

      psnr=[]
      ssim=[]
  
      for ipath1 in ipathset:
  
        for i in range(len(targets)):
          targets[i][0]=ipath1
        
        np.random.seed(123)
    
        root_dir = 'results_{}/{}/{}/{}'.format(int(round(t0)),model,tname,os.path.splitext(os.path.split(ipath1)[1])[0])
        if not os.path.exists(root_dir):
          os.makedirs(root_dir)
    
        # Generate activations for input images
        target_data_list = gen_target_data(root_dir, caffe, net, targets)
    
        # Generate white noise image
        init_img = np.random.normal(loc=0.5, scale=0.1, size=image_dims + (3,))
    
        solver_type = 'L-BFGS-B'
        solver_param = {}
    
        #test_all_gradients(init_img, net, all_target_blob_names, targets, target_data_list)
    
        A=caffe.io.load_image(ipath1)
        B=net.preprocess_inputs([A],auto_reshape=True)
        C=net.transformer.deprocess(net.inputs[0],B)
        D=caffe.io.resize_image(C,A.shape)
        print('input',A.shape,A.dtype,A.min(),A.max())
        print('pre',B.shape,B.dtype,B.min(),B.max())
        print('de',C.shape,C.dtype,C.min(),C.max())
        print('re',D.shape,D.dtype,D.min(),D.max())
    
        Chat=optimize_img(
          init_img, solver_type, solver_param, max_iter, display, root_dir, net,
          all_target_blob_names, targets, target_data_list
        )
        Dhat=caffe.io.resize_image(Chat,A.shape)
        print('best psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,D,1).mean(),measure.measure_SSIM(A,D,1).mean()))
        print('actual psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean()))
        skimage.io.imsave('{}/eval_original.png'.format(root_dir),A)
        skimage.io.imsave('{}/eval_best.png'.format(root_dir),D)
        skimage.io.imsave('{}/eval_actual.png'.format(root_dir),Dhat)
        caption='psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean())
        subprocess.check_call('convert {root_dir}/eval_best.png {root_dir}/eval_actual.png -size {w}x -font Arial-Italic -pointsize 14 caption:{caption} -append {root_dir}/eval.png'.format(root_dir=pipes.quote(root_dir),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)
        psnr.append(measure.measure_PSNR(A,Dhat,1).mean())
        ssim.append(measure.measure_SSIM(A,Dhat,1).mean())
  
      psnr=np.asarray(psnr).mean()
      ssim=np.asarray(ssim).mean()
      with open('results_{}/autoencoder.txt'.format(int(round(t0))),'a') as f:
        f.write('{},{},{},{}\n'.format(model,tname,psnr,ssim))

  t1=time.time()
  print('Finished in {} minutes.'.format((t1-t0)/60.0))

def read_lfw_attributes(ipath='dataset/lfw/lfw_attributes.txt'):
  # We verify that the first two attributes are person and sequence number
  # (from which the filename can be constructed).
  with open(ipath) as f:
    header=f.readline()
    attributes=f.readline().split('\t')[1:]
    assert attributes[0]=='person'
    assert attributes[1]=='imagenum'
    return header,attributes,[x.split('\t') for x in f.readlines()]

def lfw_filename(person,seq):
  person=person.replace(' ','_')
  return '{}/{}_{:04}.jpg'.format(person,person,int(seq))

def deepart_extract(model='vggface',blob_names=['conv3_1','conv4_1','conv5_1']):
  rlprint=ratelimit(interval=60)(print)

  _,_,lfwattr=read_lfw_attributes()
  for x in lfwattr:
    ipath='images/lfw/{}'.format(lfw_filename(x[0],x[1]))
    assert os.path.exists(ipath)
  print('lfw count =',len(lfwattr))
  caffe,net,image_dims=setup_classifier(model=model)
  h5f={}
  ds={}
    
  # minibatch processing
  M=10
  work_units,work_done,work_t0=len(lfwattr),0,time.time()
  for i,x in enumerate(minibatch(lfwattr,M)):
    inputs=['images/lfw/'+lfw_filename(y[0],y[1]) for y in x]
    F=net.extract_features(inputs,blob_names,auto_reshape=True)
    for k,v in F.items():
      if i==0:
        print(k,v.shape,v.dtype,v.min(),v.max())
        h5f[k]=h5py.File('data_{}.h5'.format(k),'w')
        ds[k]=h5f[k].create_dataset('DS',(len(lfwattr),)+v.shape[1:],chunks=(1,)+v.shape[1:],dtype='float32',compression='gzip',compression_opts=1)
        assert v.shape[0]==M
      ds[k][i*M:i*M+v.shape[0]]=v[:]
    work_done=work_done+v.shape[0]
    rlprint('{}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))

  for k in h5f:
    h5f[k].close()

  # Example reading code:
  #h5f=h5py.File('data_conv3_1.h5','r')
  #a=h5f['DS'][0]
  #print(a.shape,a.dtype,a.min(),a.max()) # should be (256,56,56)
  #h5f.close()

def deepart_reconstruct(model='vggface',blob_names=['conv3_1','conv4_1','conv5_1'],blob_weights=[1,1,1],prefix='data',subsample=1000,max_iter=100):
  t0=time.time()

  # create network
  caffe,net,image_dims=setup_classifier(model=model)

  # init result dir
  root_dir='results_{}'.format(int(round(t0)))
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  def print(*args):
    with open('{}/log.txt'.format(root_dir),'a') as f:
      f.write(' '.join(str(x) for x in args)+'\n')
    sys.stdout.write(' '.join(str(x) for x in args)+'\n')
  print('root_dir',root_dir)
  rlprint=ratelimit(interval=60)(print)

  # read features
  h5f={}
  for k in blob_names:
    assert os.path.exists('{}_{}.h5'.format(prefix,k))
    h5f[k]=h5py.File('{}_{}.h5'.format(prefix,k),'r')
    print('h5f',k,h5f[k]['DS'].shape,h5f[k]['DS'].dtype)
    N=h5f[k]['DS'].shape[0]
  _,_,lfwattr=read_lfw_attributes()
  assert len(lfwattr)==N

  # processing
  psnr=[]
  ssim=[]
  work_units,work_done,work_t0=len(lfwattr),0,time.time()
  for i,x in enumerate(lfwattr):
    if i % subsample: continue
    np.random.seed(123)

    ipath='images/lfw/'+lfw_filename(x[0],x[1])
    person=lfwattr[i][0]
    seq=lfwattr[i][1]

    # generate target list and target features
    all_target_blob_names=list(blob_names)
    targets=[]
    target_data_list=[]
    for k,v in zip(blob_names,blob_weights):
      if len(targets)>0 and targets[-1][3]==v:
        targets[-1][1].append(k)
        target_data_list[-1][k]=h5f[k]['DS'][i]
      else:
        targets.append((None,[k],False,v))
        target_data_list.append({k: h5f[k]['DS'][i]})
    #target_data_list = gen_target_data(root_dir, caffe, net, targets)

    # Set initial value and reshape net
    init_img=np.random.normal(loc=0.5,scale=0.1,size=image_dims+(3,))
    deepart.set_data(net,init_img)
    x0=np.ravel(init_img).astype(np.float64)
    bounds=zip(np.full_like(x0,-128),np.full_like(x0,128))
    solver_type='L-BFGS-B'
    solver_param={'maxiter': max_iter}
    opt_res=scipy.optimize.minimize(deepart.objective_func,x0,args=(net,all_target_blob_names,targets,target_data_list),bounds=bounds,method=solver_type,jac=True,options=solver_param)
    #print('opt_res',opt_res)
    #print('opt_res.x',opt_res.x.shape,opt_res.x.dtype)

    data=np.reshape(opt_res.x,net.get_input_blob().shape)[0]
    deproc_img=net.transformer.deprocess(net.inputs[0],data)
    A=caffe.io.load_image(ipath)
    B=np.clip(deproc_img,0,1)
    B=caffe.io.resize_image(B,A.shape)

    #print('A',A.shape,A.dtype,A.min(),A.max())
    #print('B',B.shape,B.dtype,B.min(),B.max())
    basename=os.path.splitext(os.path.split(lfw_filename(person,seq))[1])[0]
    skimage.io.imsave('{}/{}.png'.format(root_dir,basename),B)
    caption='psnr = {:.4}, ssim = {:.4}'.format(measure.measure_PSNR(A,B,1).mean(),measure.measure_SSIM(A,B,1).mean())
    subprocess.check_call('convert {ipath} {root_dir}/{basename}.png -size {w}x -font Arial-Italic -pointsize 14 caption:{caption} -append {root_dir}/eval_{basename}.png'.format(root_dir=pipes.quote(root_dir),basename=pipes.quote(basename),ipath=pipes.quote(ipath),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)
    psnr.append(measure.measure_PSNR(A,B,1).mean())
    ssim.append(measure.measure_SSIM(A,B,1).mean())
    with open('{}/results.txt'.format(root_dir),'a') as f:
      f.write('"{}",{},{},{}\n'.format(person,seq,psnr,ssim))

    work_done=work_done+1
    rlprint('{}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))
  for k in h5f:
    h5f[k].close()

  psnr=np.asarray(psnr).mean()
  ssim=np.asarray(ssim).mean()
  with open('{}/results.txt'.format(root_dir),'a') as f:
    f.write(',,{},{}\n'.format(psnr,ssim))
  print('psnr = {:.4}, ssim = {:.4}'.format(psnr,ssim))

  t1=time.time()
  print('Finished in {} minutes.'.format((t1-t0)/60.0))

if __name__ == '__main__':
  args=sys.argv[1:]

  #run_deepart(ipath1=args[0],ipath2=args[1],max_iter=int(args[2]))
  #deepart_identity()
  #deepart_extract()
  deepart_reconstruct()
