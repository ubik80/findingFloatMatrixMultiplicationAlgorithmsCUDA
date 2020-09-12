
import backpropCUDA as bp
import numpy as np

dims = np.load("dims.npy")
n = dims [0]
p = dims [1]
nn = n * n

p = p - 1

nueAB = 0.1
nueC = 0.1
tol = 0.01
maxNumOfIters = 3000000
useMasks = True

MA = np.ones([p, nn])
MB = np.ones([p, nn])
MC = np.ones([nn, p])

while True:
    minDistance = 10000.0
    iter = 0;
    while minDistance > 1000.0:
        Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wc = np.random.rand(p*nn).reshape([nn, p])*2.0-1.0
        iter = iter + 1
        minDistance = bp.multipleBackpropMasked(Wa, Wb, Wc, MA, MB, MC, maxNumOfIters, nueAB, nueC, tol, iter, 36, 32, useMasks, 10)

        if Wa.size < nn*p or Wb.size < nn*p or Wc.size < nn*p or minDistance < 0.0:
            print ("corrupted output")
            quit()

    np.save("dims", [n,p])
    np.save("solution_n" + str(n) + "_p" +str(p), [Wa,Wb,Wc.T])
    p = p - 1
