# coding: utf8
import numpy as np
import backprop as bp
import backpropCUDA as bpC
import multiprocessing as mp
import time
import uuid
from numba import jit
import os
np.set_printoptions(precision=2, suppress=True)


def checkSolution(W, limit):
    p = W[0].shape[0]
    nn = W[0].shape[1]
    n = int(np.sqrt(nn))
    Wa = np.round(W[0])
    Wb = np.round(W[1])
    Wc = np.round(W[2])
    Wa = np.maximum(np.minimum(Wa, 1.0), -1.0)
    Wb = np.maximum(np.minimum(Wb, 1.0), -1.0)
    Wc = np.maximum(np.minimum(Wc, 1.0), -1.0)
    BIdx = np.array([k*n for k in range(n)])
    c = np.zeros(nn, dtype=float)

    @jit(nopython=True, nogil=True, cache=True)
    def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc, limit):
        for i in range(100):
            a = np.random.rand(nn)*2.0-1.0
            b = np.random.rand(nn)*2.0-1.0
            nA = np.linalg.norm(a, 2)
            nB = np.linalg.norm(b, 2)

            if np.abs(nA) > 0.1 and np.abs(nB) > 0.1:
                a /= nA
                b /= nB

                for ii in range(n):  # Matrixmultiplikation für abgerollte Matrizen
                    AA = a[ii*n:ii*n+n]
                    for jj in range(n):
                        BB = b[BIdx+jj]
                        c[ii*n+jj] = AA.dot(BB)

                aWaveStar = Wa.dot(a)
                bWaveStar = Wb.dot(b)
                cWaveStar = aWaveStar*bWaveStar
                cWave = Wc.dot(cWaveStar)
                errC = cWave-c
                err2Norm = np.linalg.norm(errC, 2)
                if err2Norm > limit:
                    return False
            else:
                i -= 1
        return True  # fastLoop
    ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc, limit)
    return ret  # checkSolution


def PA(W):
    W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
    W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
    W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
    return W  # PA


def PB(W):
    nn = W[0].shape[1]
    p = W[0].shape[0]
    n = int(np.sqrt(nn))
    numOfProc = 4
    WAs = [mp.RawArray('d', W[0].reshape(nn*p)) for i in range(numOfProc)]
    WBs = [mp.RawArray('d', W[1].reshape(nn*p)) for i in range(numOfProc)]
    WCs = [mp.RawArray('d', W[2].reshape(nn*p)) for i in range(numOfProc)]

    def backprop(WaMP, WbMP, WcMP, nn, p, i):
        Wa = np.frombuffer(WaMP, dtype='d').reshape([p, nn])
        Wb = np.frombuffer(WbMP, dtype='d').reshape([p, nn])
        Wc = np.frombuffer(WcMP, dtype='d').reshape([nn, p])
        bp.backpropNotMasked(Wa, Wb, Wc, 3000000, 0.01, 0.1, 0.1, i)
        return  # backprop

    procs = [mp.Process(target=backprop, args=(WAs[i], WBs[i], WCs[i], nn, p, i))
             for i in range(numOfProc)]
    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()

    WAs = [np.frombuffer(WAs[i], dtype='d').reshape([p, nn]) for i in range(numOfProc)]
    WBs = [np.frombuffer(WBs[i], dtype='d').reshape([p, nn]) for i in range(numOfProc)]
    WCs = [np.frombuffer(WCs[i], dtype='d').reshape([nn, p]) for i in range(numOfProc)]

    minDist = 999.9
    WaRet, WbRet, WcRet = [], [], []
    success = False
    for i in range(numOfProc):
        if not (np.isnan(WAs[i].any()) or np.isnan(WBs[i].any()) or np.isnan(WCs[i].any())):
            dist = np.linalg.norm(WAs[i]-W[0], 2)**2+np.linalg.norm(WBs[i]-W[1],
                                                                    2)**2+np.linalg.norm(WCs[i]-W[2], 2)**2
            if dist < minDist:
                minDist = dist
                WaRet, WbRet, WcRet = WAs[i], WBs[i], WCs[i]
                success = True
    return [WaRet, WbRet, WcRet], success  # PB


@jit(nopython=True, nogil=True, cache=True)
def rankWeight(i, j, Wa, Wb, Wc, baseDev, matSel, WaiWci, WbiWci, WajWbj, ei):
    if matSel == 0:
        deltaVec = (np.round(Wa[i, j])-Wa[i, j])*WbiWci[i]
    elif matSel == 1:
        deltaVec = (np.round(Wb[i, j])-Wb[i, j])*WaiWci[i]
    else:
        ei *= 0.0
        ei[i] = 1.0
        deltaVec = (np.round(Wc[i, j])-Wc[i, j])*WajWbj[j]*ei
    ret = np.linalg.norm(deltaVec+baseDev, 2)
    return ret  # rankWeight


@jit(nopython=True, nogil=True, cache=True)
def findWeight(Wa, Wb, Wc, MA, MB, MC, ei):
    nn = Wa.shape[1]
    p = Wa.shape[0]
    n = int(np.sqrt(nn))
    bestI = -999
    bestJ = -999
    bestErr = 999999
    matSel = -9
    a = np.ones(nn)
    b = np.ones(nn)
    baseDev = Wc.dot(Wa.dot(a)*Wb.dot(b))-np.ones(nn)*n
    WaiWci = [np.sum(Wa[i, :])*Wc[:, i] for i in range(p)]
    WbiWci = [np.sum(Wb[i, :])*Wc[:, i] for i in range(p)]
    WajWbj = [np.sum(Wa[j, :])*np.sum(Wb[j, :]) for j in range(p)]
    for i in range(p):
        for j in range(nn):
            if MA[i, j] == 1:
                err = rankWeight(i, j, Wa, Wb, Wc, baseDev, 0, WaiWci, WbiWci, WajWbj, ei)
                if err < bestErr:
                    bestErr = err
                    bestI = i
                    bestJ = j
                    matSel = 0
    for i in range(p):
        for j in range(nn):
            if MB[i, j] == 1:
                err = rankWeight(i, j, Wa, Wb, Wc, baseDev, 1, WaiWci, WbiWci, WajWbj, ei)
                if err < bestErr:
                    bestErr = err
                    bestI = i
                    bestJ = j
                    matSel = 1
    for i in range(nn):
        for j in range(p):
            if MC[i, j] == 1:
                err = rankWeight(i, j, Wa, Wb, Wc, baseDev, 2, WaiWci, WbiWci, WajWbj, ei)
                if err < bestErr:
                    bestErr = err
                    bestI = i
                    bestJ = j
                    matSel = 2
    return bestI, bestJ, bestErr, matSel  # findWeight


def roundInit(n, p):
    nn = int(n**2)
    success = -1
    while success < 0:
        print("roundInit - Initialisierung ...")
        Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
        success = bp.backpropNotMasked(Wa, Wb, Wc, 3000000, 0.01, 0.1, 0.1, 0)
        print("roundInit - success=", success)
    print("roundInit - Initialisierung erfolgreich")
    MA = np.ones(Wa.shape)
    MB = np.ones(Wb.shape)
    MC = np.ones(Wc.shape)
    TA = np.ones(Wa.shape)
    TB = np.ones(Wb.shape)
    TC = np.ones(Wc.shape)
    ei = np.zeros(nn, dtype=float)
    rounds = 0
    noRoundForIters = 0
    iters = 0
    while True:
        i, j, err, matSel = findWeight(Wa, Wb, Wc, TA, TB, TC, ei)
        if i < 0:
            break
        WaT = Wa.copy()
        WbT = Wb.copy()
        WcT = Wc.copy()
        if matSel == 0:
            TA[i, j] = 0
            MA[i, j] = 0
            WaT[i, j] = np.minimum(np.maximum(np.round(WaT[i, j]), -1), 1)
        if matSel == 1:
            TB[i, j] = 0
            MB[i, j] = 0
            WbT[i, j] = np.minimum(np.maximum(np.round(WbT[i, j]), -1), 1)
        if matSel == 2:
            TC[i, j] = 0
            MC[i, j] = 0
            WcT[i, j] = np.minimum(np.maximum(np.round(WcT[i, j]), -1), 1)
        success = bp.backpropMasked(WaT, WbT, WcT, MA, MB, MC, 100000, 0.1, 0.1, 0.01)
        if success > 0:
            Wa = WaT
            Wb = WbT
            Wc = WcT
            rounds += 1
            noRoundForIters = 0
            print("o", end=" ", flush=True)
        else:
            if matSel == 0:
                MA[i, j] = 1
            if matSel == 1:
                MB[i, j] = 1
            if matSel == 2:
                MC[i, j] = 1
            noRoundForIters += 1
            print("x", end=" ", flush=True)
        # if noRoundForIters > 100:
        #     break
        iters += 1
    print("roundInit-Rundungen: ", str(rounds))
    return [Wa, Wb, Wc]  # roundInit


def diffMap(n, p, id):
    nn = int(n**2)
    print("n: ", n, "     p: ", p, "     beta: 1")
    seed = int(time.time())+int(uuid.uuid4())+id
    np.random.seed(seed % 135790)
    W = roundInit(n, p)
    i = 0  # iteration
    diffs = []
    maxNumIters = 5000  # fits for n=3, p=23
    jumpFactor = 0.25  # fits for n=3, p=23
    minDiff = 99999
    maxDiff = -99999
    inBand = 0
    bandWith = 10  # fits for n=3, p=23
    while True:
        s = False
        while not s:
            PBx, s = PB(W)
            if not s:
                print("   Prz: ", id, " BP failed -> reset")
                seed = int(time.time())+int(uuid.uuid4())+id
                np.random.seed(seed % 135745)
                W = roundInit(n, p)
                minDiff = 99999
                maxDiff = -99999
                inBand = 0
                i = 0
        PAy = PA([2.0*PBx[0]-W[0], 2.0*PBx[1]-W[1], 2.0*PBx[2]-W[2]])
        delta = [PAy[0]-PBx[0], PAy[1]-PBx[1], PAy[2]-PBx[2]]
        W = [W[0]+delta[0], W[1]+delta[1], W[2]+delta[2]]
        norm2Delta = np.linalg.norm(
            delta[0], 2)**2+np.linalg.norm(delta[1], 2)**2+np.linalg.norm(delta[2], 2)**2
        norm2Delta = np.sqrt(norm2Delta)
        diffs.append(norm2Delta)

        if norm2Delta < 0.5:
            print(id, ", Lösung gefunden?")
            WW = PA(PB(W)[0])
            if checkSolution(WW,  0.00000001):
                print(id, ".... Lösung korrekt")
                np.save("solution", [WW[0], WW[1], WW[2]])
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
            W[0] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*jumpFactor
            W[1] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*jumpFactor
            W[2] += (np.random.rand(p*nn).reshape([nn, p])*2.0-1.0)*jumpFactor
            inBand = 0
        if i > maxNumIters:
            print(i, " cycles -> Reset")
            seed = int(time.time())+int(uuid.uuid4())+id
            np.random.seed(seed % 135790)
            W = roundInit(n, p)
            minDiff = 99999
            maxDiff = -99999
            inBand = 0
            i = 0
        i += 1
    return  # diffMap


if __name__ == '__main__':
    diffMap(n=3, p=23, id=0)
