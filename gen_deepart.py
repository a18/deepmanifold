#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

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

from fet_extractor import load_fet_extractor
from deepart import gen_target_data, optimize_img
from test_deepart import test_all_gradients
import measure
import deepart
import matchmmd

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

def filter_args(args,valid_args,help_args,depth=1):
  caller_globals=inspect.stack()[depth][0].f_globals
  if '--help' in args:
    for x in valid_args:
      print('--{:30} (Default: {}){}'.format(x,caller_globals[x],'' if x not in help_args else ' '+help_args[x]))
    sys.exit(1)
  result=[]
  for x in args:
    if not x.startswith('--'):
      result.append(x)
      continue
    if '=' not in x:
      k=x[2:]
      if k not in valid_args:
        print('Unknown option {}'.format('--'+k))
        sys.exit(1)
      caller_globals[k]=True
    else:
      k,v=x.split('=',1)
      k=k[2:]
      if k not in valid_args:
        print('Unknown option {}'.format('--'+k))
        sys.exit(1)
      if isinstance(caller_globals[k],str):
        caller_globals[k]=v
      elif isinstance(caller_globals[k],tuple):
        try:
          caller_globals[k]=tuple(ast.literal_eval(v))
        except:
          # fallback to yaml, it can handle strings without quotes
          caller_globals[k]=tuple(yaml.load(v))
      elif isinstance(caller_globals[k],list):
        try:
          caller_globals[k]=list(ast.literal_eval(v))
        except:
          # fallback to yaml, it can handle strings without quotes
          caller_globals[k]=list(yaml.load(v))
      else:
        try:
          caller_globals[k]=ast.literal_eval(v) # QQQ big problem here: literal_eval performs arithmetic e.g., 2011-11-11 becomes the integer 1989
        except:
          # this does not handle nested strings properly
          caller_globals[k]=v
  return result

def setup_classifier(model='vgg',image_dims=(224,224),device_id=0):
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
    caption='psnr {:.4}, ssim {:.4}'.format(measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean())
    subprocess.check_call('convert {root_dir}/eval_best.png {root_dir}/eval_actual.png -size {w}x -font Arial-Italic -pointsize 12 caption:{caption} -append {root_dir}/eval.png'.format(root_dir=pipes.quote(root_dir),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)

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

def plot_horizontal_bars(X,Y,xlabel,ylabel,title):
  sns=seaborn
  plt=matplotlib.pyplot

  sns.set(style="whitegrid")
  sns.set_color_codes("pastel")

  f,ax=plt.subplots(figsize=(6,0.8*len(Y)))
  data=pandas.DataFrame(data={xlabel: X, ylabel: Y})
  data.plot(ylabel,xlabel,kind='barh',color='b',figsize=(6,0.2*len(Y)),edgecolor='b')

  #ax.legend(ncol=2,loc='lower right',frameon=True)
  #ax.set(xlim=(min(X),max(X)),ylabel=ylabel,xlabel=xlabel)
  sns.despine(left=True,bottom=True)

  matplotlib.pyplot.title(title)
  matplotlib.pyplot.tight_layout()

def deepart_examine(model='vgg',image_dims=None,device_id=0,max_iter=3000,hybrid_names=[],hybrid_weights=[],tv_lambda=0.001,tv_beta=2,desc='identity',dataset='lfw_random',count=20,layers=None):
  t0=time.time()

  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)

  importance={}
  for ipath3 in ['results_1448431008_identity_lfw20c2c3c4c5/vgg/c4','results_1448431008_identity_lfw20c2c3c4c5/vgg/c5']:
  #for ipath3 in ['results_1448673175_identity_houzz20c4/vgg/c4']:
    for ipath1 in glob.glob('{}/*_original.png'.format(ipath3)):
      print(ipath1)
      ipath2=ipath1.replace('original','actual')
  
      #A=caffe.io.load_image(ipath1) # ground truth
      #B=net.preprocess_inputs([A],auto_reshape=True)
      #C=net.transformer.deprocess(net.inputs[0],B)
      #D=caffe.io.resize_image(C,A.shape) # best possible result (only preprocess / deprocess)
      #Dhat=caffe.io.load_image(ipath2) # recon result
  
      # conv1_1 (2, 64, 800, 800) float32 -0.0 29.7672
      # conv2_1 (2, 128, 400, 400) float32 -0.0 35.3852
      # conv3_1 (2, 256, 200, 200) float32 -0.0 45.8031
      # conv4_1 (2, 512, 100, 100) float32 -0.0 24.7847
      # conv5_1 (2, 512, 50, 50) float32 -0.0 45.1556
  
      blob_names=['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
      F=net.extract_features([ipath1,ipath2],list(blob_names),auto_reshape=True)
      for k in blob_names:
        #print(k,F[k].shape,F[k].dtype,F[k].min(),F[k].max())
        K=F[k].shape[1]
        x=F[k][0].reshape(F[k].shape[1],-1)
        y=F[k][1].reshape(F[k].shape[1],-1)
        err=((y-x)**2).mean(axis=1)
        ranking=sorted(list(range(K)),key=lambda x: err[x])
        print(k,ranking[:10],ranking[-10:])
        if k not in importance:
          importance[k]=np.zeros(K,np.int64)
        for i in ranking[:10]:
          importance[k][i]=importance[k][i]+1
  
    for k in blob_names:
      print(k)
      K=len(importance[k])
      ranking=sorted(list(range(K)),key=lambda x: importance[k][x])
      plot_horizontal_bars([importance[k][i] for i in ranking],ranking,'top-10 error count','feature id',k)
      #fig=matplotlib.pyplot.figure()
      #matplotlib.pyplot.gcf().set_size_inches(0.1*K,4)
      #matplotlib.pyplot.bar(range(K),[importance[k][i] for i in ranking])
      #matplotlib.pyplot.xticks(range(K),ranking)
      #matplotlib.pyplot.title(k)
      #matplotlib.pyplot.xlabel('feature id')
      #matplotlib.pyplot.ylabel('top-10 error count')
      ##matplotlib.pyplot.tight_layout(pad=1)
      matplotlib.pyplot.savefig('{}/{}.pdf'.format(ipath3,k))
      matplotlib.pyplot.close()
      for i in ranking:
        print('{:4}'.format(i),'*'*importance[k][i])
    #subprocess.check_call('pdftk {}/conv*.pdf cat output {}/top-10-error.pdf'.format(pipes.quote(ipath3),pipes.quote(ipath3)),shell=True)
    subprocess.check_call('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile={}/top-10-error.pdf {}/conv*.pdf'.format(pipes.quote(ipath3),pipes.quote(ipath3)),shell=True)

  t1=time.time()
  print('Finished in {} minutes.'.format((t1-t0)/60.0))

def deepart_identity(image_dims=None,max_iter=3000,hybrid_names=[],hybrid_weights=[],tv_lambda=0.001,tv_beta=2,desc='identity',device_id=0,dataset='lfw_random',count=20,layers=None):
  # Experimenting with making deepart produce the identity function
  t0=time.time()

  # init result dir
  root_dir='results_{}'.format(int(round(t0))) if desc=='' else 'results_{}_{}'.format(int(round(t0)),desc)
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  def print(*args):
    with open('{}/log.txt'.format(root_dir),'a') as f:
      f.write(' '.join(str(x) for x in args)+'\n')
    sys.stdout.write(' '.join(str(x) for x in args)+'\n')

  print('image_dims',image_dims)
  print('max_iter',max_iter)
  print('hybrid_names',hybrid_names)
  print('hybrid_weights',hybrid_weights)
  print('tv_lambda',tv_lambda)
  print('tv_beta',tv_beta)
  print('desc',desc)
  print('device_id',device_id)
  print('dataset',dataset)
  print('count',count)
  print('layers',layers)

  if isinstance(dataset,list) or isinstance(dataset,tuple):
    ipathset=list(dataset)
  else:
    with open('dataset/{}.txt'.format(dataset)) as f:
      ipathset=['images/'+x.strip() for x in f.readlines()]
    ipathset=ipathset[:count]

  if layers is None:
    targetset=[
      ('c5',['conv5_1'],[1]),
      ('c4',['conv4_1'],[1]),
      ('c3',['conv3_1'],[1]),
      ('c2',['conv2_1'],[1]),
    ]
  else:
    targetset=[]
    if 'c2' in layers: targetset.append(('c2',['conv2_1'],[1]))
    if 'c3' in layers: targetset.append(('c3',['conv3_1'],[1]))
    if 'c4' in layers: targetset.append(('c4',['conv4_1'],[1]))
    if 'c5' in layers: targetset.append(('c5',['conv5_1'],[1]))

  #modelset=['vggface','vgg']
  modelset=['vgg']

  for model in modelset:

    caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)

    for tname,blob_names,blob_weights in targetset:

      psnr=[]
      ssim=[]
  
      for ipath1 in ipathset:
  
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

        # evaluate
        print('{} best psnr = {:.4}, ssim = {:.4}'.format(basename,measure.measure_PSNR(A,D,1).mean(),measure.measure_SSIM(A,D,1).mean()))
        print('{} actual psnr = {:.4}, ssim = {:.4}'.format(basename,measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean()))
        skimage.io.imsave('{}/{}_original.png'.format(root_dir2,basename),A)
        skimage.io.imsave('{}/{}_best.png'.format(root_dir2,basename),D)
        skimage.io.imsave('{}/{}_actual.png'.format(root_dir2,basename),Dhat)
        caption='psnr {:.4}, ssim {:.4}'.format(measure.measure_PSNR(A,Dhat,1).mean(),measure.measure_SSIM(A,Dhat,1).mean())
        subprocess.check_call('convert {root_dir2}/{basename}_original.png {root_dir2}/{basename}_actual.png -size {w}x -font Arial-Italic -pointsize 12 caption:{caption} -append {root_dir2}/eval_{basename}.png'.format(root_dir2=pipes.quote(root_dir2),basename=pipes.quote(basename),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)
        psnr.append(measure.measure_PSNR(A,Dhat,1).mean())
        ssim.append(measure.measure_SSIM(A,Dhat,1).mean())
  
      print('psnr',psnr)
      print('ssim',ssim)
      psnr=np.asarray(psnr).mean()
      ssim=np.asarray(ssim).mean()
      with open('{}/autoencoder.txt'.format(root_dir),'a') as f:
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

def deepart_extract(ipath,prefix='data',model='vgg',blob_names=['conv3_1','conv4_1','conv5_1'],image_dims=(224,224),device_id=0):
  # ipath = text file listing one image per line
  # model = vgg | vggface
  # blob_names = list of blobs to extract
  rlprint=ratelimit(interval=60)(print)

  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)
  h5f={}
  ds={}
    
  # minibatch processing
  M=10
  with open(ipath) as f:
    S=[x.strip() for x in f.readlines()]
  print('count =',len(S))
  work_units,work_done,work_t0=len(S),0,time.time()
  for i,x in enumerate(minibatch(S,M)):
    inputs=[os.path.join('images',y) for y in x]
    F=net.extract_features(inputs,blob_names,auto_reshape=True)
    for k,v in F.items():
      if i==0:
        print(k,v.shape,v.dtype,v.min(),v.max())
        h5f[k]=h5py.File('{}_{}.h5'.format(prefix,k),'w')
        ds[k]=h5f[k].create_dataset('DS',(len(S),)+v.shape[1:],chunks=(1,)+v.shape[1:],dtype='float32',compression='gzip',compression_opts=1)
        assert v.shape[0]==min(M,len(S))
      ds[k][i*M:i*M+v.shape[0]]=v[:]
    work_done=work_done+v.shape[0]
    rlprint('{}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))

  for k in h5f:
    h5f[k].close()

def deepart_extractlfw(model='vgg',blob_names=['conv3_1','conv4_1','conv5_1'],image_dims=(224,224)):
  # model = vgg | vggface
  # blob_names = list of blobs to extract
  rlprint=ratelimit(interval=60)(print)

  _,_,lfwattr=read_lfw_attributes()
  for x in lfwattr:
    ipath='images/lfw/{}'.format(lfw_filename(x[0],x[1]))
    assert os.path.exists(ipath)
  print('lfw count =',len(lfwattr))
  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims)
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

def non_local_means(ipath,w,n,h,opath):
  a=skimage.io.imread(ipath)/255.0
  b=skimage.restoration.nl_means_denoising(a,w,n,h)
  skimage.io.imsave(opath,b)
  return b

def deepart_edit(model='vgg',blob_names=['conv3_1','conv4_1','conv5_1'],blob_weights=[1,1,1],prefix='data',subsample=1,max_iter=2000,test_indices=None,data_indices=None,image_dims=(224,224),device_id=0,hybrid_names=[],hybrid_weights=[],tv_lambda=0.001,tv_beta=2,gaussian_init=False,dataset='lfw',desc='edit'):
  t0=time.time()

  # create network
  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)

  # init result dir
  root_dir='results_{}'.format(int(round(t0))) if desc=='' else 'results_{}_{}'.format(int(round(t0)),desc)
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  def print(*args):
    with open('{}/log.txt'.format(root_dir),'a') as f:
      f.write(' '.join(str(x) for x in args)+'\n')
    sys.stdout.write(' '.join(str(x) for x in args)+'\n')
  print('root_dir',root_dir)
  print('model',model)
  print('blob_names',blob_names)
  print('blob_weights',blob_weights)
  print('hybrid_names',hybrid_names)
  print('hybrid_weights',hybrid_weights)
  print('prefix',prefix)
  print('subsample',subsample)
  print('max_iter',max_iter)
  print('image_dims',image_dims)
  print('tv_lambda',tv_lambda)
  print('tv_beta',tv_beta)
  print('gaussian_init',gaussian_init)
  print('dataset',dataset)

  # image
  ipath='images/lfw/Winona_Ryder/Winona_Ryder_0024.jpg'
  init_img=caffe.io.load_image(ipath)
  print('init_img',init_img.shape,init_img.dtype)

  # generate target list and target features
  all_target_blob_names=list(hybrid_names)+list(blob_names)
  targets=[]
  target_data_list=[]
  if len(hybrid_weights)>0:
    F=net.extract_features([ipath],hybrid_names,auto_reshape=True)
    for k,v in zip(hybrid_names,hybrid_weights):
      if len(targets)>0 and targets[-1][3]==v:
        targets[-1][1].append(k)
        target_data_list[-1][k]=F[k]
      else:
        targets.append((None,[k],False,v))
        target_data_list.append({k: F[k]})
      print('hybrid',k,v,F[k].shape,F[k].dtype)
  for k,v in zip(blob_names,blob_weights):
    F=net.extract_features([ipath],blob_names,auto_reshape=True)
    if len(targets)>0 and targets[-1][3]==v:
      targets[-1][1].append(k)
      target_data_list[-1][k]=F[k]
    else:
      targets.append((None,[k],False,v))
      target_data_list.append({k: F[k]})
    print('target',k,v,F[k].shape,F[k].dtype)

  # image target = weighted L2 loss (1 x 3 x H x W)
  # gradient target = weighted L2 loss on finite diff (2 x 1 x K x H x W)
  # feature target = weighted L2 loss (1 x K x H x W)
  gradient_space_targets=[]
  if False:
    deepart.set_data(net,init_img)
    gen_data=net.get_input_blob().astype(np.float64)
    gradient_target=np.zeros((2,)+gen_data.shape,dtype=np.float64)
    gradient_target[0,:,:,:-1,:]=np.diff(gen_data,axis=2)*3
    gradient_target[1,:,:,:,:-1]=np.diff(gen_data,axis=3)*3
    gradient_weight=np.ones(gen_data.shape)
    gradient_space_targets.append((gradient_target,gradient_weight))

  image_space_targets=[]
  if False:
    color_img=skimage.io.imread('eyes.png')/255.0
    deepart.set_data(net,color_img[:,:,:3])
    gen_data=net.get_input_blob().astype(np.float64)
    image_target=np.copy(gen_data)
    image_weight=(color_img[:,:,3])[np.newaxis,np.newaxis]
    assert image_target.shape==(1,3,250,250)
    assert image_weight.shape==(1,1,250,250)
    assert image_weight.max()<=1
    image_space_targets.append((image_target,image_weight))

  if True:
    F=net.extract_features([ipath],all_target_blob_names,auto_reshape=True)
    k='conv3_1'
    v=1
    #print(k,F[k].min(),F[k].max(),np.linalg.norm(F[k]))
    for i in range(F[k].shape[1]):
      m=F[k][0,i].max()
      F[k][0,i]*=2
    targets.append((None,[k],False,v))
    target_data_list.append({k: np.copy(F[k])})
  

  # objective fn
  def objective_fn(x, net, all_target_blob_names, targets, target_data_list, tv_lambda, tv_beta):
    # def objective_func(x, net, all_target_blob_names, targets, target_data_list, tv_lambda, tv_beta):
    # x = current solution image
    # returns loss, gradients
    deepart.get_data_blob(net).data[...]=np.reshape(x,deepart.get_data_blob(net).data.shape)
    deepart.get_data_blob(net).diff[...]=0
    net.forward()

    loss = 0
    # Go through target blobs in reversed order
    for i in range(len(all_target_blob_names)):
        blob_i = len(all_target_blob_names) - 1 - i
        start = all_target_blob_names[blob_i]

        if blob_i == 0:
            end = None
        else:
            end = all_target_blob_names[blob_i - 1]

        # Get target blob
        target_blob = net.blobs[start]
        if i == 0:
            target_blob.diff[...] = 0

        gen_data = target_blob.data.copy()
        print('gen_data',gen_data.shape,gen_data.dtype) # debug
        # Apply RELU
        pos_mask = gen_data > 0
        gen_data[~pos_mask] = 0

        # Go through all images and compute accumulated gradient for the current target blob
        target_blob_add_diff = np.zeros_like(target_blob.diff, dtype=np.float64)
        for target_i, (_, target_blob_names, is_gram, weight) in enumerate(targets):
            # Skip if the current blob is not among the target's blobs
            if start not in target_blob_names:
                continue

            target_data = target_data_list[target_i][start]
            if is_gram:
                c_loss, c_grad = deepart.style_grad(gen_data, target_data)
            else:
                c_loss, c_grad = deepart.content_grad(gen_data, target_data)

            # Apply RELU
            c_grad[~pos_mask] = 0
            target_blob_add_diff += c_grad * weight / len(target_blob_names)
            loss += c_loss * weight / len(target_blob_names)

        target_blob.diff[...] += target_blob_add_diff
        net.backward(start=start, end=end)

    print('loss',loss)
    grad = np.ravel(deepart.get_data_blob(net).diff).astype(np.float64)

    # debug
    for (gradient_target, gradient_weight) in gradient_space_targets:
      gen_data = x.reshape(deepart.get_data_blob(net).data.shape)
      fy = np.diff(gen_data, axis=2)
      fx = np.diff(gen_data, axis=3)
      loss_g, grad_g = deepart.gradient_grad(gen_data, gradient_target, gradient_weight)
      grad_g = np.ravel(grad_g).astype(np.float64)
      loss += loss_g
      grad += grad_g
      print('loss_g',loss_g)

    for (image_target, image_weight) in image_space_targets:
      loss_i, grad_i = deepart.content_grad(gen_data, image_target, weight=image_weight)
      grad_i = np.ravel(grad_i).astype(np.float64)
      loss += loss_i
      grad += grad_i
      print('loss_i',loss_i)

    if tv_lambda > 0:
        tv_loss, tv_grad = totalvariation.tv_norm(x.reshape(deepart.get_data_blob(net).data.shape),beta=tv_beta)
        print('loss_tv',tv_loss*tv_lambda)
        return loss + tv_loss*tv_lambda, grad + np.ravel(tv_grad)*tv_lambda
    else:
        return loss, grad

  deepart.set_data(net,init_img)
  x0=net.get_input_blob().ravel().astype(np.float64)
  bounds=zip(np.full_like(x0,-128),np.full_like(x0,162))
  solver_type='L-BFGS-B'
  solver_param={'maxiter': max_iter, 'iprint': -1}
  opt_res=scipy.optimize.minimize(objective_fn,x0,args=(net,all_target_blob_names,targets,target_data_list,tv_lambda,tv_beta),bounds=bounds,method=solver_type,jac=True,options=solver_param)
  print(opt_res)

  data=np.reshape(opt_res.x,net.get_input_blob().shape)[0]
  deproc_img=net.transformer.deprocess(net.inputs[0],data)
  B=np.clip(deproc_img,0,1)
  A=init_img

  print('A',A.shape,A.dtype,A.min(),A.max())
  print('B',B.shape,B.dtype,B.min(),B.max())
  skimage.io.imsave('{}/input.png'.format(root_dir),A)
  skimage.io.imsave('{}/output.png'.format(root_dir),B)

  t1=time.time()
  print('Finished in {} minutes.'.format((t1-t0)/60.0))

def deepart_reconstruct(model='vgg',blob_names=['conv3_1','conv4_1','conv5_1'],blob_weights=[1,1,1],prefix='data',subsample=1,max_iter=2000,test_indices=None,data_indices=None,image_dims=(224,224),device_id=0,hybrid_names=[],hybrid_weights=[],tv_lambda=0.001,tv_beta=2,gaussian_init=False,dataset='lfw',desc=''):
  # model = vgg | vggface
  # blob_names = list of blobs to match (must be in the right order, front to back)
  # blob_weights = cost function weight for each blob
  # prefix = target features will be read from PREFIX_BLOB.h5
  # subsample = process every N from the dataset
  # max_iter = number of iters to optimize (2000+ for good quality)
  # test_indices = list of dataset indices (corresponds to each entry in h5 files)
  # data_indices = list of h5 indices (for computing subsets of the data)
  #   Example: data_indices=[0,3], test_indices=[4,2] means compute with the first
  #   and fourth features in the h5 file and compare against the fifth and third
  #   images in the dataset.

  t0=time.time()

  # create network
  caffe,net,image_dims=setup_classifier(model=model,image_dims=image_dims,device_id=device_id)

  # init result dir
  root_dir='results_{}'.format(int(round(t0))) if desc=='' else 'results_{}_{}'.format(int(round(t0)),desc)
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  def print(*args):
    with open('{}/log.txt'.format(root_dir),'a') as f:
      f.write(' '.join(str(x) for x in args)+'\n')
    sys.stdout.write(' '.join(str(x) for x in args)+'\n')
  print('root_dir',root_dir)
  print('model',model)
  print('blob_names',blob_names)
  print('blob_weights',blob_weights)
  print('hybrid_names',hybrid_names)
  print('hybrid_weights',hybrid_weights)
  print('prefix',prefix)
  print('subsample',subsample)
  print('max_iter',max_iter)
  print('image_dims',image_dims)
  print('tv_lambda',tv_lambda)
  print('tv_beta',tv_beta)
  print('gaussian_init',gaussian_init)
  print('dataset',dataset)
  rlprint=ratelimit(interval=60)(print)

  # read features
  h5f={}
  for k in blob_names:
    assert os.path.exists('{}_{}.h5'.format(prefix,k))
    h5f[k]=h5py.File('{}_{}.h5'.format(prefix,k),'r')
    print('h5f',k,h5f[k]['DS'].shape,h5f[k]['DS'].dtype)
    N=h5f[k]['DS'].shape[0]
  #_,_,lfwattr=read_lfw_attributes()
  with open('dataset/{}.txt'.format(dataset)) as f:
    original_names=[x.strip() for x in f.readlines()]
  if data_indices is None:
    # assume you want to process everything
    data_indices=list(range(N))
  else:
    # require that you specify the data -> dataset mapping
    assert not test_indices is None
    assert len(data_indices)==len(test_indices)
  if test_indices is None:
    test_indices=list(range(N))

  for x in hybrid_names:
    assert x not in blob_names
  assert len(hybrid_names)==len(hybrid_weights)

  # processing
  psnr=[]
  ssim=[]
  work_units,work_done,work_t0=len(test_indices),0,time.time()
  basename_uid={}
  for j,i in enumerate(test_indices):
    if j % subsample: continue
    np.random.seed(123)

    #ipath='images/lfw/'+lfw_filename(lfwattr[i][0],lfwattr[i][1])
    ipath='images/'+original_names[i]
    #person=lfwattr[i][0]
    #seq=lfwattr[i][1]
    #basename=os.path.splitext(os.path.split(lfw_filename(person,seq))[1])[0]
    basename=os.path.splitext(os.path.split(ipath)[1])[0]
    if basename not in basename_uid:
      basename_uid[basename]=0
    else:
      basename_uid[basename]=basename_uid[basename]+1
    basename2='{}-{:02}'.format(basename,basename_uid[basename])

    # generate target list and target features
    all_target_blob_names=list(hybrid_names)+list(blob_names)
    targets=[]
    target_data_list=[]
    if len(hybrid_weights)>0:
      F=net.extract_features([ipath],hybrid_names,auto_reshape=True)
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
        target_data_list[-1][k]=h5f[k]['DS'][data_indices[j]]
      else:
        targets.append((None,[k],False,v))
        target_data_list.append({k: h5f[k]['DS'][data_indices[j]]})
      print('target',k,v,h5f[k]['DS'][data_indices[j]].shape,h5f[k]['DS'][data_indices[j]].dtype)
    #target_data_list = gen_target_data(root_dir, caffe, net, targets)

    # Set initial value and reshape net
    if gaussian_init:
      init_img=np.random.normal(loc=0.5,scale=0.1,size=image_dims+(3,))
    else:
      init_img=caffe.io.load_image(ipath)
    deepart.set_data(net,init_img)
    #x0=np.ravel(init_img).astype(np.float64)
    x0=net.get_input_blob().ravel().astype(np.float64)
    bounds=zip(np.full_like(x0,-128),np.full_like(x0,162))
    solver_type='L-BFGS-B'
    solver_param={'maxiter': max_iter, 'iprint': -1}
    opt_res=scipy.optimize.minimize(deepart.objective_func,x0,args=(net,all_target_blob_names,targets,target_data_list,tv_lambda,tv_beta),bounds=bounds,method=solver_type,jac=True,options=solver_param)
    #print('opt_res',opt_res)
    #print('opt_res.x',opt_res.x.shape,opt_res.x.dtype)

    data=np.reshape(opt_res.x,net.get_input_blob().shape)[0]
    deproc_img=net.transformer.deprocess(net.inputs[0],data)
    A=caffe.io.load_image(ipath)
    B=np.clip(deproc_img,0,1)
    A=caffe.io.resize_image(A,B.shape)

    #print('A',A.shape,A.dtype,A.min(),A.max())
    #print('B',B.shape,B.dtype,B.min(),B.max())
    skimage.io.imsave('{}/{}-original.png'.format(root_dir,basename),A)
    skimage.io.imsave('{}/{}.png'.format(root_dir,basename2),B)
    #C=non_local_means('{}/{}.png'.format(root_dir,basename2),3,21,0.04,'{}/{}-nlm.png'.format(root_dir,basename2))
    caption='psnr {:.4}, ssim {:.4}'.format(measure.measure_PSNR(A,B,1).mean(),measure.measure_SSIM(A,B,1).mean())
    subprocess.check_call('convert {root_dir}/{basename}-original.png {root_dir}/{basename2}.png -size {w}x -font Arial-Italic -pointsize 12 caption:{caption} -append {root_dir}/eval_{basename2}.png'.format(root_dir=pipes.quote(root_dir),basename=pipes.quote(basename),basename2=pipes.quote(basename2),ipath=pipes.quote(ipath),caption=pipes.quote(caption),w=A.shape[1],h=A.shape[0]//10),shell=True)
    psnr.append(measure.measure_PSNR(A,B,1).mean())
    ssim.append(measure.measure_SSIM(A,B,1).mean())
    with open('{}/results.txt'.format(root_dir),'a') as f:
      f.write('"{}",{},{},{}\n'.format(basename2,i,psnr[-1],ssim[-1]))

    work_done=work_done+1*subsample
    rlprint('{}/{}, {} min remaining'.format(work_done,work_units,(work_units/work_done-1)*(time.time()-work_t0)/60.0))
  for k in h5f:
    h5f[k].close()

  print('psnr',psnr)
  print('ssim',ssim)
  psnr=np.asarray(psnr).mean()
  ssim=np.asarray(ssim).mean()
  with open('{}/results.txt'.format(root_dir),'a') as f:
    f.write(',{},{}\n'.format(psnr,ssim))

  t1=time.time()
  print('Finished in {} minutes.'.format((t1-t0)/60.0))

def attr_pairs(attr,index,k1,k2):
  # returns top-k strongest and weakest image indices
  i=list(range(len(attr)))
  if index<0:
    i.sort(key=lambda x: float(attr[x][-index]))
  else:
    i.sort(key=lambda x: -float(attr[x][index]))
  return i[:k1],(i[-k2:] if k2>0 else [])

def deepart_match(prefix='data',desc='match',blob_names=['conv3_1','conv4_1','conv5_1'],weights=[1e-5,7.5e-6,5e-6],attr=10,source_k=2000,target_k=2000,test_indices=[0,1,2,3,4],image_dims=(224,224),device_id=0):
  t0=time.time()

  # init result dir
  root_dir='results_{}'.format(int(round(t0))) if desc=='' else 'results_{}_{}'.format(int(round(t0)),desc)
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  def print(*args):
    with open('{}/log.txt'.format(root_dir),'a') as f:
      f.write(' '.join(str(x) for x in args)+'\n')
    sys.stdout.write(' '.join(str(x) for x in args)+'\n')
  print('root_dir',root_dir)
  print('prefix',prefix)
  print('desc',desc)
  print('blob_names',blob_names)
  print('weights',weights)
  print('attr',attr)
  print('source_k',source_k)
  print('target_k',target_k)
  print('test_indices',test_indices)

  # read pca matrices
  data=np.load('{}_pca.npz'.format(prefix))
  U=data['U']
  T=data['T']
  mu=data['mu']
  shape=data['shape'].item()
  del data
  print('U',U.shape,U.dtype,U.min(),U.max())
  print('T',T.shape,T.dtype,T.min(),T.max())
  print('mu',mu.shape,mu.dtype,mu.min(),mu.max())
  print('shape',shape)

  # setup source and target distributions
  _,_,lfwattr=read_lfw_attributes()
  if attr>=0:
    target_indices,source_indices=attr_pairs(lfwattr,attr,target_k,source_k)
  else:
    source_indices,target_indices=attr_pairs(lfwattr,-attr,source_k,target_k)
  print('source_indices',source_indices)
  print('target_indices',target_indices)
  P=T[source_indices]
  Q=T[target_indices]
  print('P',P.shape,P.dtype)
  print('Q',Q.shape,Q.dtype)

  # match target
  allF=[]
  for i in test_indices:
    x_0,x,r=matchmmd.match_distribution(T[i],P,Q,weights)
    print('test_index',i)
    print('x_0',x_0.shape,x_0.dtype,x_0.min(),x_0.max())
    print('x',x.shape,x.dtype,x.min(),x.max())
    print('r',r.shape,r.dtype,r.min(),r.max())
    F=x.dot(U)+mu
    print('F',F.shape,F.dtype,F.min(),F.max())
    allF.append(F)

  F=np.asarray(allF,dtype=np.float32)
  F=F.reshape(F.shape[0]*F.shape[1],-1)
  print('F',F.shape,F.dtype,F.min(),F.max())
  index=0
  for k in blob_names:
    size=np.prod(shape[k][1:])
    h5f=h5py.File('{}/{}_{}.h5'.format(root_dir,desc,k),'w')
    h5f.create_dataset('DS',data=F[:,index:index+size].reshape(*shape[k]))
    h5f.close()
    index=index+size

  del U
  del T
  del F

  deepart_reconstruct(blob_names=blob_names,blob_weights=[1]*len(blob_names),prefix=root_dir+'/'+desc,max_iter=3000,test_indices=list(np.repeat(test_indices,len(weights))),data_indices=None,image_dims=image_dims,hybrid_names=['conv1_1','conv2_1'],hybrid_weights=[0.02,0.02],dataset='lfw',desc=desc+'_reconstruct',device_id=device_id)

  print('Finished in {} minutes.'.format((time.time()-t0)/60.0))

class EconomyPCA:
  def __init__(self):
    pass
  def fit_transform(self,F):
    # F is N x P
    print('F',F.shape,F.dtype,F.min(),F.max(),(F==0).sum(),((F!=0).sum(axis=0)==0).sum())
    t0=time.time()
    FFT=np.dot(F,F.T)
    print('F.F^T in {} minutes.'.format((time.time()-t0)/60.0))
    sys.stdout.flush()
    t0=time.time()
    a,V=np.linalg.eigh(FFT)
    print('a',a.shape,a.dtype,a.min(),a.max())
    print('V',V.shape,V.dtype)
    print('eigh in {} minutes.'.format((time.time()-t0)/60.0))
    sys.stdout.flush()
    t0=time.time()
    W=np.dot(F.T,V)
    print('W',W.shape,W.dtype)
    print('F^T.V in {} minutes.'.format((time.time()-t0)/60.0))
    sys.stdout.flush()
    t0=time.time()
    b=np.sqrt(np.abs(a))*np.sign(a)
    W=W/b
    print('b',b.shape,b.dtype)
    print('W',W.shape,W.dtype)
    G=np.dot(F,W)
    print('G',G.shape,G.dtype)
    self.W=W
    print('normalization and transform in {} minutes.'.format((time.time()-t0)/60.0))
    sys.stdout.flush()
    return G
  def inverse_transform(self,G):
    return np.dot(G,self.W.T)

def deepart_pca(prefix='data',blob_names=['conv3_1','conv4_1','conv5_1'],method='economy',dtype='float32'):
  t0=time.time()
  print('method',method)
  print('blob_names',blob_names)
  print('prefix',prefix)

  # read features
  h5f={}
  F=[]
  shape={}
  for k in blob_names:
    assert os.path.exists('{}_{}.h5'.format(prefix,k))
    h5f[k]=h5py.File('{}_{}.h5'.format(prefix,k),'r')
    print('h5f',k,h5f[k]['DS'].shape,h5f[k]['DS'].dtype)
    N=h5f[k]['DS'].shape[0]
    shape[k]=(-1,)+tuple(h5f[k]['DS'].shape[1:])
    F.append(np.asarray(h5f[k]['DS'],dtype=dtype).reshape(N,-1))
  for k in h5f:
    h5f[k].close()
  del h5f
  F=np.concatenate(F,axis=1)
  #F=F[::10,::10] # debug
  N=F.shape[0]
  print('Memory: {} GB'.format(resource.getrusage(resource.RUSAGE_SELF)[2]*1024/(2**30)))
  print('N',N)
  print('F',F.shape,F.dtype,F.min(),F.max(),(F==0).sum(),((F!=0).sum(axis=0)==0).sum())
  print('shape',shape)

  mu=F.mean(axis=0)
  print('mu',mu.shape,mu.dtype,mu.min(),mu.max())
  sys.stdout.flush()
  if method=='matlab':
    # First, run deepart_pca.m
    # Then, run this code (which merely saves it as npz)
    h5f=h5py.File('pca_U.h5','r')
    U=np.copy(h5f['DS'])
    h5f.close()
    print('U',U.shape,U.dtype,U.min(),U.max())
    h5f=h5py.File('pca_T.h5','r')
    T=np.copy(h5f['DS'])
    h5f.close()
    print('T',T.shape,T.dtype,T.min(),T.max())
    h5f=h5py.File('pca_mu.h5','r')
    mu=np.copy(h5f['DS'])
    h5f.close()
    print('mu',mu.shape,mu.dtype,mu.min(),mu.max())
    F2=T.dot(U)+mu
    print('F2',F2.shape,F2.dtype,F2.min(),F2.max())
    np.savez('{}_pca.npz'.format(prefix),U=U,T=T,mu=mu,shape=shape)
    print('Wrote {}_pca.npz'.format(prefix))
  elif method=='economy':
    pca=EconomyPCA()
    pca.mu=mu
    G=pca.fit_transform(F-pca.mu)
    F2=pca.inverse_transform(G)+pca.mu
    print('G',G.shape,G.dtype,G.min(),G.max())
    print('F2',F2.shape,F2.dtype,F2.min(),F2.max())
    np.savez('{}_pca.npz'.format(prefix),U=pca.W.T,T=G,mu=pca.mu,shape=shape)
    print('Wrote {}_pca.npz'.format(prefix))
  else:
    raise ValueError

  # Example reading code:
  # data=np.load('data_pca.npz')
  # shape=data['shape'].item()

  t1=time.time()
  print('Memory: {} GB'.format(resource.getrusage(resource.RUSAGE_SELF)[2]*1024/(2**30)))
  print('Finished in {} minutes.'.format((t1-t0)/60.0))

def deepart_compare(inputs,name='compare'):
  t0=time.time()
  imshape=skimage.io.imread(glob.glob('{}/eval_*'.format(inputs[0]))[0]).shape
  inputs=[y for x in inputs for y in glob.glob('{}/eval_*'.format(x))]
  print(inputs)
  root_dir='results_{}'.format(int(round(t0)))
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  subprocess.check_call('montage -geometry {}x{}+6+6 {} {}/{}.png'.format(imshape[1],imshape[0],' '.join(pipes.quote(x) for x in inputs),root_dir,name),shell=True)
  print('{}/{}.png'.format(root_dir,name))

if __name__ == '__main__':
  args=sys.argv[1:]

  #run_deepart(ipath1=args[0],ipath2=args[1],max_iter=int(args[2]))
  if args[0]=='identity':
    args=args[1:]
    image_dims=None
    device_id=0
    dataset='lfw_random'
    count=20
    desc='identity'
    layers=None
    params=('image_dims','device_id','dataset','count','desc','layers')
    params_desc={}
    args=filter_args(args,params,params_desc)
    deepart_identity(image_dims=image_dims,device_id=device_id,dataset=dataset,count=count,desc=desc,layers=layers)
  elif args[0]=='examine':
    args=args[1:]
    model='vgg'
    image_dims=None
    device_id=0
    params=('model','image_dims','device_id')
    params_desc={}
    args=filter_args(args,params,params_desc)
    deepart_examine(model=model,image_dims=image_dims,device_id=device_id)
  elif args[0]=='extractlfw':
    args=args[1:]
    model='vgg'
    image_dims=(125,125)
    params=('model','image_dims')
    params_desc={'model': 'vgg | vggface'}
    args=filter_args(args,params,params_desc)
    deepart_extractlfw(model=model,image_dims=image_dims)
  elif args[0]=='extract':
    args=args[1:]
    model='vgg'
    image_dims=(125,125)
    prefix='data'
    device_id=0
    params=('model','image_dims','prefix','device_id')
    params_desc={'model': 'vgg | vggface', 'prefix': 'Prefix for output files'}
    args=filter_args(args,params,params_desc)
    deepart_extract(args[0],model=model,image_dims=image_dims,prefix=prefix,device_id=device_id)
  elif args[0]=='pca':
    args=args[1:]
    method='economy'
    blob_names=['conv3_1','conv4_1','conv5_1']
    #blob_names=['conv5_1'] # debug
    prefix='data'
    dtype='float32'
    params=('method','blob_names','prefix','dtype')
    params_desc={'method': 'matlab | economy', 'dtype': 'float32 | float64', 'prefix': 'Prefix for output file and feature vector files'}
    args=filter_args(args,params,params_desc)
    deepart_pca(method=method,blob_names=blob_names,prefix=prefix)
  elif args[0]=='match':
    args=args[1:]
    prefix='data'
    desc='match'
    blob_names=['conv3_1','conv4_1','conv5_1']
    weights=[1e-5,7.5e-6,5e-6]
    attr=10
    source_k=2000
    target_k=2000
    test_indices=[6005, 3659, 8499, 12217, 9982, 4322, 10449, 10969, 4245, 7028] # randomly selected
    image_dims=(125,125)
    device_id=0
    params=('prefix','desc','blob_names','weights','attr','source_k','target_k','test_indices','image_dims','device_id')
    params_desc={'prefix': 'Prefix for input files', 'desc': 'Will be appended to the results directory name', 'weights': 'Budget of change parameter', 'attr': 'Target attribute (or source attribute if negative)', 'source_k': 'Number of images in source set', 'target_k': 'Number of images in target set', 'test_indices': 'Zero-indexed list of images to optimize'}
    args=filter_args(args,params,params_desc)
    deepart_match(prefix=prefix,desc=desc,blob_names=blob_names,weights=weights,attr=attr,source_k=source_k,target_k=target_k,test_indices=test_indices,image_dims=image_dims,device_id=device_id)
  elif args[0]=='reconstruct':
    args=args[1:]
    model='vgg'
    test_indices=None
    data_indices=None
    subsample=1
    max_iter=3000
    image_dims=(125,125)
    prefix='data'
    device_id=0
    blob_names=['conv3_1','conv4_1','conv5_1']
    blob_weights=[1,1,1]
    hybrid_names=['conv1_1','conv2_1']
    hybrid_weights=[0.02,0.02]
    tv_lambda=0.001
    tv_beta=2
    gaussian_init=False
    dataset='lfw'
    desc=''
    params=('model','test_indices','data_indices','subsample','max_iter','image_dims','prefix','device_id','blob_names','blob_weights','hybrid_names','hybrid_weights','tv_lambda','tv_beta','gaussian_init','dataset','desc')
    params_desc={'model': 'vgg | vggface', 'test_indices': 'which dataset images to compare against', 'data_indices': 'which entries in the h5 files to compute', 'hybrid_names': 'Must be in the same order as in the network', 'tv_lambda': 'Total variation loss weight', 'dataset': 'Original image filenames read from dataset/DATASET.txt', 'desc': 'Will be appended to the results directory name'}
    args=filter_args(args,params,params_desc)
    deepart_reconstruct(model=model,test_indices=test_indices,data_indices=data_indices,subsample=subsample,max_iter=max_iter,image_dims=image_dims,prefix=prefix,device_id=device_id,blob_names=blob_names,blob_weights=blob_weights,hybrid_names=hybrid_names,hybrid_weights=hybrid_weights,tv_lambda=tv_lambda,tv_beta=tv_beta,gaussian_init=gaussian_init,dataset=dataset,desc=desc)
  elif args[0]=='compare':
    args=args[1:]
    deepart_compare(inputs=args)
  elif args[0]=='edit':
    args=args[1:]
    model='vgg'
    test_indices=None
    data_indices=None
    subsample=1
    max_iter=1
    image_dims=(250,250)
    prefix='data'
    device_id=0
    blob_names=['conv3_1','conv4_1','conv5_1']
    blob_weights=[1,1,1]
    hybrid_names=['conv1_1','conv2_1']
    hybrid_weights=[0.02,0.02]
    tv_lambda=0.001
    tv_beta=2
    gaussian_init=False
    dataset='lfw'
    desc='edit'
    params=('model','test_indices','data_indices','subsample','max_iter','image_dims','prefix','device_id','blob_names','blob_weights','hybrid_names','hybrid_weights','tv_lambda','tv_beta','gaussian_init','dataset','desc')
    params_desc={'model': 'vgg | vggface', 'test_indices': 'which dataset images to compare against', 'data_indices': 'which entries in the h5 files to compute', 'hybrid_names': 'Must be in the same order as in the network', 'tv_lambda': 'Total variation loss weight', 'dataset': 'Original image filenames read from dataset/DATASET.txt', 'desc': 'Will be appended to the results directory name'}
    args=filter_args(args,params,params_desc)
    deepart_edit(model=model,test_indices=test_indices,data_indices=data_indices,subsample=subsample,max_iter=max_iter,image_dims=image_dims,prefix=prefix,device_id=device_id,blob_names=blob_names,blob_weights=blob_weights,hybrid_names=hybrid_names,hybrid_weights=hybrid_weights,tv_lambda=tv_lambda,tv_beta=tv_beta,gaussian_init=gaussian_init,dataset=dataset,desc=desc)
  elif args[0]=='compare':
    args=args[1:]
    deepart_compare(inputs=args)
  else:
    raise ValueError('Unknown command')

