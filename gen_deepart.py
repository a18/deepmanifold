import numpy as np
import sys
import skimage.io
import time
import os
import os.path
import subprocess
import pipes

from fet_extractor import load_fet_extractor
from deepart import gen_target_data, optimize_img
from test_deepart import test_all_gradients
import measure


def setup_classifier(model='vgg',image_dims=(224,224),device_id=1):
    #deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_deepart.prototxt'
    #weights_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers.caffemodel'
    #image_dims = (1014/2, 1280/2)
    #mean = (104, 117, 123)

    if model=='vgg':
        deployfile_relpath = 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'
        weights_relpath = 'models/VGG_CNN_19/vgg_normalised.caffemodel'
        mean = (103.939, 116.779, 123.68)
    elif model=='vggface':
        deployfile_relpath = 'models/vgg_face_caffe/VGG_FACE_deploy_conv.prototxt'
        weights_relpath = 'models/vgg_face_caffe/VGG_FACE.caffemodel'
        mean = (103.939, 116.779, 123.68)
    else:
        raise ValueError('Unknown CNN model:',model)
    input_scale = 1.0

    caffe, net = load_fet_extractor(
        deployfile_relpath, weights_relpath, image_dims, mean, device_id,
        input_scale
    )

    return caffe, net, image_dims


def deepart(ipath1='images/starry_night.jpg',ipath2='images/tuebingen.jpg',max_iter=2000):
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

def deepart_autoencoder(max_iter=1000):
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

  targetset=[
    #('c5',[ ['', ['conv5_1'], False, 1], ]),
    ('c4',[ ['', ['conv4_1'], False, 1], ]),
    ('c3',[ ['', ['conv3_1'], False, 1], ]),
    #('c45',[ ['', ['conv4_1', 'conv5_1'], False, 1], ]),
    #('c345',[ ['', ['conv3_1', 'conv4_1', 'conv5_1'], False, 1], ]),
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
  print 'Finished in {} minutes.'.format((t1-t0)/60.0)

if __name__ == '__main__':
    args=sys.argv[1:]

    #deepart(ipath1=args[0],ipath2=args[1],max_iter=int(args[2]))
    deepart_autoencoder()

