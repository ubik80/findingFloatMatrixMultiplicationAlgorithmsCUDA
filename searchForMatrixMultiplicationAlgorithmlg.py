
import sys
import backpropCUDA as bp # pybind11 module in Cuda
import numpy as np

dims = np.load("dims.npy") # load the dimensions of the problem to solve
n = dims [0] # matrix dimensions of A and B
p = dims [1] # number of products
nn = n * n
p = p - 1 # problem already solved for p, now solve for p - 1

numOfBlocks = 36; # number of CUDA blocks to use
numOfThreads = 64; # number of threads per block

nueAB = 0.1 # learning rate for Wa and Wb
nueC = 0.1 # learning rate for Wc
tol = 0.01 # tolerance for error in C (2-norm)
maxNumOfIters = 3000000 # max number of iterations for backpropCUDA

Wa = np.ndarray(p * nn).reshape([p, nn]) # result is written here
Wb = np.ndarray(p * nn).reshape([p, nn])
Wc = np.ndarray(p * nn).reshape([nn, p])

while True: # endlessly search for even better algorithms (lower p)
    error = sys.float_info.max
    iter = 0;
    while error > sys.float_info.max / 2.0: # while no solution found
        iter = iter + 1
        # calculate solution on GPU
        error = bp.backpropCUDA(Wa, Wb, Wc,
            maxNumOfIters, nueAB, nueC, tol, iter, numOfBlocks, numOfThreads)
        # check if result is plausible
        if Wa.size < nn * p or Wb.size < nn * p or Wc.size < nn * p or error < 0.0 or error > tol * 1.1:
            print ("corrupted output")
            error = sys.float_info.max

    # check result through calculation
    for i in range(100):
        a = np.random.rand(nn) * 2.0 - 1.0
        b = np.random.rand(nn) * 2.0 - 1.0
        a = a / np.linalg.norm(a, 2)
        b = b / np.linalg.norm(b, 2)
        c = Wc.dot((Wa.dot(a) * Wb.dot(b)))
        A = a.reshape([n, n])
        B = b.reshape([n, n])
        C = A.dot(B)
        cWave = C.reshape(nn)
        err = np.linalg.norm(c - cWave, 2)
        if err > tol * 1.1:
            print ("err > tol, err = " + str(err))
            quit()

    np.save("dims", [n, p]) 
    np.save("solution_n" + str(n) + "_p" + str(p), [Wa, Wb, Wc.T]) # save solution
    p = p - 1 # repeat with less products ...
    Wa = np.ndarray(p * nn).reshape([p, nn])
    Wb = np.ndarray(p * nn).reshape([p, nn])
    Wc = np.ndarray(p * nn).reshape([nn, p])
