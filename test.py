
import backpropCUDA as bp
import numpy as np

p=4*4*4-1
n=4
nn=n*n

nueAB = 0.1
nueC = 0.1
tol = 0.01
maxNumOfIters = 3000000;

MA = np.ones([p, nn])
MB = np.ones([p, nn])
MC = np.ones([nn, p])

success = 1.0
iter = 0;
while success > tol:
    Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wc = np.random.rand(p*nn).reshape([nn, p])*2.0-1.0
    iter = iter+1
    success = bp.multipleBackpropMasked(Wa, Wb, Wc, MA, MB, MC, maxNumOfIters, nueAB, nueC, tol, iter, 4, 32)
    print (success)
