
import sys
import backpropCUDA as bp # pybind11 module in CUDA
import numpy as np

dims = np.load("dims.npy") # load the dimensions of the problem to solve
n = dims [0] # matrix dimensions of A and B
p = dims [1] # number of products
nn = n * n
p = p - 1 # problem already solved for p, now solve for p - 1

Wa = np.random.rand(p * nn).reshape([p, nn]) * 2.0 - 1.0
Wb = np.random.rand(p * nn).reshape([p, nn]) * 2.0 - 1.0
Wc = np.random.rand(p * nn).reshape([nn, p]) * 2.0 - 1.0

minDistance = bp.multipleBackpropMasked(Wa, Wb, Wc, 0.0, 0.0, 0.0,
                    3000000, 0.1, 0.1, 0.01, 42, 36, 64, False, 5)
print(minDistance)
