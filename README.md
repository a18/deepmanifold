# Deep Manifold Traversal

## Deep Manifold Traversal (2016 March)

See http://arxiv.org/abs/1511.06421 for an explanation of the method.

Run `./scripts/install/install_all.sh` to install. This will install python dependencies, download caffe, compile caffe, and download the necessary weight file. It is assumed that caffe lives at `../caffe` and it will not be downloaded if it already exists.

The main method is `dmt.run()` which takes a list of image pathnames where the first N entries define the source submanifold, the next M entries define the target submanifold, the next L entries (optional) are additional images which increase the number of basis vectors and the remaining entries are the images to be transformed. See the function documentation for details.

```python
ipath = [source1, ..., sourceN, target1, ..., targetM, data1, ..., dataL, image1, ...]
weights = [lambda1, lambda2, ...]
XF,F2,root_dir,result = dmt.run(ipath,N,M,L,'vgg',[125,125],device_id,weights,rbf_var,prefix,3000,False,False)
```

The `run_dmt.py` script is a minimal wrapper which demonstrates how to call `dmt.run()`. Look inside `run_dmt.py` for hints on how to set lambda weights and rbf kernel variances.



## PCA deep manifold traversal (2015 November)

The old `gen_deepart.py` methods are now deprecated. The newer method is faster and uses less memory.
