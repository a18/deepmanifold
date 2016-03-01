# Deep Manifold Traversal

## PCA deep manifold traversal

See http://arxiv.org/abs/1511.06421 for an explanation of the method.

Run `./scripts/install/install_all.sh` to install. This will install python dependencies, download caffe, compile caffe, and download the necessary weight file. It is assumed that caffe lives at `../caffe` and it will not be downloaded if it already exists.

The primary script is `gen_deepart.py`. `./gen_deepart.py COMMAND --help` will list command-line options for any of the following commands: extract, pca, match and reconstruct.

Run `./gen_deepart.py extract dataset/lfw.txt` to extract ConvNet features for the LFW dataset.

Run `./gen_deepart.py pca` to compute the PCA projection. Note: you will need 85 GB (128 GB for float64) of memory.

Run `./gen_deepart.py match` to produce a synthetic aging demonstration. The match command will move a test image (`--test_indices`) away from a source distribution and toward a target distribution. The size of each distribution and the selected attribute are set by `--source_k`, `--target_k`, and `--attr` (e.g., 10 for Senior). Look inside `dataset/lfw_attributes.txt` for a list of attributes. The latent vector results are stored in a directory named `results_TIME_match` where `TIME` is seconds after the UNIX epoch. Reconstructed images are stored in `results_TIME_match_reconstruct`.

The preceeding command reconstructs images with default reconstruction settings. You can run the reconstruct command directly to set different parameters. Example: ``./gen_deepart.py reconstruct --test_indices='[6005, 3659, 8499, 12217, 9982, 4322, 10449, 10969, 4245, 7028]' --prefix=results_1450738103_match/match --desc=match_reconstruct`` is equivalent to the default reconstruction.

## Experimental, no-PCA method

`dmt.py` uses a newer method which does not require running PCA on the entire dataset. This makes it possible to run on larger datasets. Call `dmt.run()` with a list of image pathnames where the first N entires are the source domain, the next M entires are the target domain and the remaining entries are the images to be transformed. See the function documentation for details.

```python
ipath = [file1, file2, ...]
weights = [lambda1, lambda2, ...]
XF,F2,root_dir,result = dmt.run(ipath,N,M,'vgg',[125,125],device_id,weights,rbf_var,prefix,3000,False)
```

Note: Different datasets may require different lambda weights and rbf kernel variances.
