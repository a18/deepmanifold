#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

# HOWTO add a custom model:
# 
# 1. Define a function, import_caffe, which returns an object which acts
# like the caffe module. See fet_extractor.py
# 
# 2. Define a function, extractor, which builds a class when given your
# caffe-like object. The returned class should act like FeatureExtractor.
# See fet_extractor.py
# 
# 3. Create an entry in the table below for your model.

def import_caffe():
  import utils
  utils.add_caffe_to_path()
  import caffe
  return caffe

from fet_extractor import def_FeatureExtractor

modeldef={
  'vgg': {
    'import_caffe': import_caffe,
    'extractor': def_FeatureExtractor,
    'deployfile_relpath': 'models/VGG_CNN_19/VGG_ILSVRC_19_layers_deploy_fullconv.prototxt',
    'weights_relpath': 'models/VGG_CNN_19/vgg_normalised.caffemodel',
    'mean': (103.939, 116.779, 123.68)
  },
  'vggface': {
    'import_caffe': import_caffe,
    'extractor': def_FeatureExtractor,
    'deployfile_relpath': 'models/vgg_face_caffe/VGG_FACE_deploy_conv.prototxt',
    'weights_relpath': 'models/vgg_face_caffe/VGG_FACE.caffemodel',
    'mean': (93.5940, 104.7624, 129.1863)
  },
}

