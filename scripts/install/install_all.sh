#!/usr/bin/env bash

set -e

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR="$DIR/../../"
cd $ROOT_DIR

echo "Installing deepart..."
./scripts/install/install_python.sh
cd ..
if [ ! -d "caffe" ]; then
	echo "Cloning caffe repo..."
	git clone https://github.com/BVLC/caffe.git
fi
cd caffe
if [ ! -f "Makefile.config" ]; then
	cp Makefile.config.example Makefile.config
fi
echo "Compiling caffe..."
make all -j8
make pycaffe

cd $ROOT_DIR
echo "Copying model file ..."
cp -r models/VGG_CNN_19 ../caffe/models/

cd $ROOT_DIR
if [ ! -f "../caffe/models/VGG_CNN_19/vgg_normalised.caffemodel" ]; then
  echo "Downloading weight file ..."
  wget http://bethgelab.org/media/uploads/deeptextures/vgg_normalised.caffemodel -P ../caffe/models/VGG_CNN_19/
fi

cd $ROOT_DIR/images
if [ ! -d "lfw" ]; then
  echo "Downloading LFW images ..."
  wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
  tar xzf lfw.tgz
  rm lfw.tgz
fi

cd $ROOT_DIR
echo "Done."

