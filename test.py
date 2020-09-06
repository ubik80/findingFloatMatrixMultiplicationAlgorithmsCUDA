
import backpropCUDA as bp
import numpy as np

p=23
n=4
nn=n*n

MA = np.ones([p, nn]).astype(np.float32)
MB = np.ones([p, nn]).astype(np.float32)
MC = np.ones([nn, p]).astype(np.float32)

Wa = np.random.rand(p*nn).astype(np.float32).reshape([p, nn])*2.0-1.0
Wb = np.random.rand(p*nn).astype(np.float32).reshape([p, nn])*2.0-1.0
Wc = np.random.rand(p*nn).astype(np.float32).reshape([nn, p])*2.0-1.0

nueAB = np.float32(0.1)
nueC = np.float32(0.1)
tol = np.float32(0.01)
maxNumOfIters = 100;

bp.multipleBackpropMasked(Wa, Wb, Wc, MA, MB, MC, maxNumOfIters, nueAB, nueC, tol, 1234)
