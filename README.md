# deepart

Run `./scripts/install/install_all.sh` to install. It will install python dependencies, download caffe, compile caffe, and download the necessary weight file.

Run `./gen_deepart.py extract dataset/lfw.txt` to extract ConvNet features for the LFW dataset.

Run `./gen_deepart.py pca` to compute the PCA projection. Note: you will need 85 GB (128 GB for float64) of memory.

Run `./gen_deepart.py match` to produce a synthetic aging demonstration. The match command will move a test image (`--test_indices`) away from a source distribution and toward a target distribution. The size of each distribution and the selected attribute are set by `--source_k`, `--target_k`, and `--attr`. The match result is stored in a directory named `results_TIME_match` where `TIME` is seconds after the UNIX epoch.
