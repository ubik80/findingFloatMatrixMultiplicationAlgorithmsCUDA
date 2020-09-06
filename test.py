
import backpropCUDA
import numpy as np

p=23
n=4
nn=n*n

Wa = np.random.rand(p*nn).astype(np.float16).reshape([p, nn])*2.0-1.0
Wb = np.random.rand(p*nn).astype(np.float16).reshape([p, nn])*2.0-1.0
Wc = np.random.rand(p*nn).astype(np.float16).reshape([nn, p])*2.0-1.0
