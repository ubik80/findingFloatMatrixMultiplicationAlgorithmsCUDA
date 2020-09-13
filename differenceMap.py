# coding: utf8
import numpy as np
import backpropCUDA as bpC
import multiprocessing as mp
import time
import uuid
import os
import sys
np.set_printoptions(precision=2, suppress=True)

n=3;
nn=n*n;
p=23

def checkSolution(W, tol):
    for i in range(100):
        a = np.random.rand(nn) * 2.0 - 1.0
        b = np.random.rand(nn) * 2.0 - 1.0
        a = a / np.linalg.norm(a, 2)
        b = b / np.linalg.norm(b, 2)
        c = W[2].dot((W[0].dot(a) * W[1].dot(b)))
        A = a.reshape([n, n])
        B = b.reshape([n, n])
        C = A.dot(B)
        cWave = C.reshape(nn)
        err = np.linalg.norm(c - cWave, 2)
        if err > tol * 1.1:
            print ("err > tol, err = " + str(err))
            return False
    return True  # checkSolution

def PA(Wa, Wb, Wc):
    print("PA")
    Wa = np.minimum(np.maximum(np.round(Wa), -1.0), 1.0)
    Wb = np.minimum(np.maximum(np.round(Wb), -1.0), 1.0)
    Wc = np.minimum(np.maximum(np.round(Wc), -1.0), 1.0)
    return Wa, Wb, Wc  # PA


def PB(Wa, Wb, Wc):
    print("PB start")
    minDistance = sys.float_info.max
    while minDistance > sys.float_info.max / 2.0: # while no solution found
        minDistance = bpC.multipleBackpropMasked(Wa, Wb, Wc, 0.0, 0.0, 0.0,
                                    3000000, 0.1, 0.1, 0.01, 42, 36, 64, False, 5)
    print("PB finished")
    return Wa, Wb, Wc  # PB

def initW(n, p):
    print("Initialisierung start")
    nn = int(n**2)

    # minDistance = sys.float_info.max
    # while minDistance > sys.float_info.max / 2.0: # while no solution found
    #     Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    #     Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    #     Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
        #
        # minDistance = bpC.multipleBackpropMasked(Wa, Wb, Wc, 0.0, 0.0, 0.0,
        #                             3000000, 0.1, 0.1, 0.01, 42, 36, 64, False, 1)
        #
        # Wa[0,0]=1.0;
        # Wb[1,1]=1.0;
        # Wc[2,2]=1.0;
        #
        # minDistance = bpC.multipleBackpropMasked(Wa, Wb, Wc, 0.0, 0.0, 0.0,
        #             3000000, 0.1, 0.1, 0.01, 42, 36, 64, False, 1)

    Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
    print("W initialized")
    return Wa, Wb, Wc

def diffMap(n, p):
    nn = int(n**2)
    print("n: ", n, "     p: ", p, "     beta: 1")
    seed = int(time.time())+int(uuid.uuid4())
    np.random.seed(seed % 135790)
    Wa, Wb, Wc = initW(n, p)


    i = 0  # iteration
    diffs = []
    maxNumIters = 5000  # fits for n=3, p=23
    jumpFactor = 0.25  # fits for n=3, p=23
    minDiff = 99999
    maxDiff = -99999
    inBand = 0
    bandWith = 10  # fits for n=3, p=23

    while True:
        WaPBx, WbPBx, WcPBx = PB(Wa.copy(), Wb.copy(), Wc.copy())
        WaPAy, WbPAy, WcPAy = PA(2.0*WaPBx-Wa, 2.0*WbPBx-Wb, 2.0*WcPBx-Wc)
        deltaA, deltaB, deltaC = WaPAy-WaPBx, WbPAy-WbPBx, WcPAy-WcPBx
        Wa, Wb, Wc = Wa+deltaA, Wb+deltaB, Wc+deltaC
        norm2Delta = np.linalg.norm(
            deltaA, 2)**2+np.linalg.norm(deltaA, 2)**2+np.linalg.norm(deltaC, 2)**2
        norm2Delta = np.sqrt(norm2Delta)
        diffs.append(norm2Delta)

        if norm2Delta < 0.5:
            print(id, ", Lösung gefunden?")
            WW = PA(PB(Wa,Wb,Wc)[0])
            if checkSolution(WW,  0.00000001):
                print(id, ".... Lösung korrekt")
                np.save("solution", [Wa, Wb, Wc])
                return
            else:
                print(id, ".... keine gültige Lösung")
        if i % 1 == 0 and i > 0:
            print("---------------------------")
            print("Iter.:  ", i)
            print("Delta:  ", norm2Delta)
        if len(diffs) > bandWith:
            minDiff = min(diffs[max(len(diffs)-bandWith, 0): len(diffs)])
            maxDiff = max(diffs[max(len(diffs)-bandWith, 0): len(diffs)])
        if norm2Delta > minDiff and norm2Delta < maxDiff:
            inBand += 1
        else:
            inBand = 0
        if inBand > bandWith:
            Wa += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*jumpFactor
            Wb += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*jumpFactor
            Wc += (np.random.rand(p*nn).reshape([nn, p])*2.0-1.0)*jumpFactor
            inBand = 0
        if i > maxNumIters:
            print(i, " cycles -> Reset")
            seed = int(time.time())+int(uuid.uuid4())+id
            np.random.seed(seed % 135790)
            W = initW(n, p)
            minDiff = 99999
            maxDiff = -99999
            inBand = 0
            i = 0
        i += 1
    return  # diffMap


if __name__ == '__main__':
    diffMap(n=2, p=7)
