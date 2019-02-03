# Compound Density Networks

## General information
1. The codes are meant to be run on a GPU.
2. The default arguments for each codes are already set so that running the codes without argument will replicate the results shown in the paper.


## Instruction
1. Install the dependencies contained in `requirements.txt`. Remember to install pytorch with GPU support, manually if necessary.
2. Create new folder called `data` and run `extract_features_cifar10.py`.
3. Run the code on a GPU, e.g.: `CUDA_VISIBLE_DEVICES=0 python ml_cdn_mnist.py`.
4. Trained models will be saved in `models/{dataset}` directory.
5. Experiment results will be saved in `results/{dataset}` directory in Numpy format, i.e. use `np.load` to load the results.
