## Deep Manifold Traversal

Deep manifold traversal transforms an image so that it is more similar to a target submanifold and less similar to a source manifold. For example, if the source and target submanifolds are images of young and old faces, then a face image can be transformed to look older. Here is how it works:

First, deep convolutional features are extracted from each image by a GPU. Next, the manifold traversal equation is optimized by a CPU to find modified features. Finally, the image is optimized by a GPU to fit the modified features.

See http://arxiv.org/abs/1511.06421 for an explanation of the method.

#### Try it out

Run `./scripts/install/install_all.sh` to install. This will install python dependencies, download caffe, compile caffe, and download the necessary weight file. It is assumed that caffe lives at `../caffe` and it will not be downloaded if it already exists. You may also need: `sudo apt-get install imagemagick`.

The following command will generate a test image:

`./run_dmt.py --source images/young/*.jpg --target images/old/*.jpg --image images/Aaron_Eckhart_0001.jpg`

#### How to write your own scripts

The main method is `dmt.run()` which takes a list of image pathnames where the first N entries define the source submanifold, the next M entries define the target submanifold, the next L entries (optional) are additional images which increase the number of basis vectors and the remaining entries are the images to be transformed. See the function documentation for details.

```python
ipath = [source1, ..., sourceN, target1, ..., targetM, data1, ..., dataL, image1, ...]
weights = [lambda1, lambda2, ...]
XF,F2,root_dir,result = dmt.run(ipath,N,M,L,'vgg',[125,125],device_id,weights,rbf_var,prefix,3000,False,False)
```

The `run_dmt.py` script is a minimal wrapper which demonstrates how to call `dmt.run()`. Look inside `run_dmt.py` for hints on how to set lambda weights and rbf kernel variances. 

#### How to process large images

We found the limit to be main memory. Amazon EC2 offers CPU machines with 244 GB of memory which is enough to process 900 x 600 images. You can optimize the manifold traversal equation there, then transfer the results to a GPU machine for the reconstruction optimization. See the `run_daynight.py` script for an example of how we do this.

#### Updates

The old `gen_deepart.py` methods (2015 November) are now deprecated. The new `dmt.py` (2016 March) is faster and uses less memory.
