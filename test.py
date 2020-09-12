
import sys
import backpropCUDA as bp # pybind11 module in CUDA
import numpy as np

dims = np.load("dims.npy") # load the dimensions of the problem to solve
n = dims [0] # matrix dimensions of A and B
p = dims [1] # number of products
nn = n * n

p = p - 1 # problem already solved for p, now solve for p - 1

numOfBlocks = 36; # number of CUDA blocks to use
numOfThreads = 32; # number of threads per block

nueAB = 0.1 # learning rate for Wa and Wb
nueC = 0.1 # learning rate for Wc
tol = 0.01 # tolerance for error in C (2-norm)
maxNumOfIters = 3000000 # max number of iterations for backpropCUDA
useMasks = False # false - ignore
minDistanceOutOf = 10 # pool of solutions from which to choose the one with min.
# distance between initial values and solution

MA = np.ones([p, nn]) # Mask for Ma, 0.0 - don't change, 1.0 - can be changed
MB = np.ones([p, nn])
MC = np.ones([nn, p])

while True: # endlessly search for even better algorithms (lower p)
    minDistance = sys.float_info.max
    iter = 0;
    while minDistance > sys.float_info.max / 2.0: # while no solution found
        Wa = np.random.rand(p * nn).reshape([p, nn]) * 2.0 - 1.0
        Wb = np.random.rand(p * nn).reshape([p, nn]) * 2.0 - 1.0
        Wc = np.random.rand(p * nn).reshape([nn, p]) * 2.0 - 1.0
        iter = iter + 1

        # calculate solution on GPU
        minDistance = bp.multipleBackpropMasked(Wa, Wb, Wc, MA, MB, MC,
            maxNumOfIters, nueAB, nueC, tol, iter, numOfBlocks, numOfThreads,
            useMasks, minDistanceOutOf)

        # check if result is plausible
        if Wa.size < nn * p or Wb.size < nn * p or Wc.size < nn * p or minDistance <= 0.0:
            print ("corrupted output")
            quit()

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

    np.save("dims", [n, p]) # reduce p to increase problem
    np.save("solution_n" + str(n) + "_p" + str(p), [Wa, Wb, Wc.T]) # save solution
    p = p - 1 # repeat with less products ...
