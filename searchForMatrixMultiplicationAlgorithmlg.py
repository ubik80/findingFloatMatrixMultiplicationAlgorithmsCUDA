import sys
import backpropCUDA as bp # pybind11 module in Cuda
import numpy as np
import time as tm

dims = np.load("dims.npy") # load the dimensions of the problem to solve
n = dims[0] # matrix dimensions of A and B
p = dims[1] # number of products
nn = n * n

numOfBlocks = 46 # number of CUDA blocks to use
numOfThreads = 128 #64 # number of threads per block

nueAB = 0.1 # learning rate for Wa and Wb
nueC = 0.1 # learning rate for Wc
tol = 0.01 # tolerance for error in C (2-norm)
maxNumOfIters = 3000000 # max number of iterations for backpropCUDA

while True: # endlessly search for even better algorithms (lower p)
    Wa = np.ndarray(p * nn, dtype=np.float32).reshape([p, nn])  # result is written here
    Wb = np.ndarray(p * nn, dtype=np.float32).reshape([p, nn])
    Wc = np.ndarray(p * nn, dtype=np.float32).reshape([nn, p])

    error = sys.float_info.max
    iter = 0;
    while error > sys.float_info.max / 4.0: # while no solution found
        iter = iter + 1
        seed = int((iter+tm.time()) % sys.maxsize)
        # calculate solution on GPU
        with np.errstate(over="ignore"):
            error = bp.backpropCUDA(Wa, Wb, Wc, maxNumOfIters, nueAB, nueC, tol, seed, numOfBlocks, numOfThreads)
        # check if result is plausible
        if Wa.size < nn * p or Wb.size < nn * p or Wc.size < nn * p or error < 0.0:
            print (f"corrupted output, error {error}")
            error = sys.float_info.max
        if error > tol * 1.1:
            print (f"error too large, error {error}")
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
        if err > tol * 2.0:
            print("err > tol, err = " + str(err))
            quit()

    np.save("dims", [n, p])
    np.save("solution_n" + str(n) + "_p" + str(p), [Wa, Wb, Wc.T]) # save solution
    p -= 1 # repeat with less products ...