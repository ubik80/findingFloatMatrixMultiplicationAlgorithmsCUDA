#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

void checkForCudaError(int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
    fprintf(stderr, "  --- in line %i:\t%s\n", line, cudaGetErrorString(err));
}

__device__ void lock(int *mutex) {
  while (atomicCAS(mutex, 0, 1) != 0) {
  };
}

__device__ void unlock(int *mutex) { atomicExch(mutex, 0); }

// memory allocation, remember pointer for cleanup later
__device__ float *mallocGb(int numOfFloats, float **garbageDump,
                           int garbageCounter) {
  garbageDump[garbageCounter] = (float *)malloc(numOfFloats * sizeof(float));
  garbageCounter++;
  return garbageDump[garbageCounter - 1];
}

// free all allocated
__device__ void freeGb(float **garbageDump, int garbageCounter) {
  for (int i = 0; i < garbageCounter; i++) {
    free(garbageDump[i]);
  }
}

__global__ void kernel(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                       float *Mc, int maxNumOfIters, float nueAB, float nueC,
                       float tol, int n, int p, int seed, float *minDistance,
                       int *mutex, int *killSignal, bool useMasks,
                       int minDistanceOutOf, int *distanceCount) {
  float *garbageDump[10];
  int garbageCounter = 0;

  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;

  const int nn = n * n;
  float *myWa = (float *)mallocGb(nn * p, garbageDump, garbageCounter);
  float *myWb = (float *)mallocGb(nn * p, garbageDump, garbageCounter);
  float *myWc = (float *)mallocGb(nn * p, garbageDump, garbageCounter);
  memcpy(myWa, Wa, nn * p * sizeof(float));
  memcpy(myWb, Wb, nn * p * sizeof(float));
  memcpy(myWc, Wc, nn * p * sizeof(float));
  float *a = (float *)mallocGb(nn, garbageDump, garbageCounter);
  float *b = (float *)mallocGb(nn, garbageDump, garbageCounter);
  float *c = (float *)mallocGb(nn, garbageDump, garbageCounter);
  float *aStar = (float *)mallocGb(p, garbageDump, garbageCounter);
  float *bStar = (float *)mallocGb(p, garbageDump, garbageCounter);
  float *cStar = (float *)mallocGb(p, garbageDump, garbageCounter);
  float *cDiff = (float *)mallocGb(nn, garbageDump, garbageCounter);

  int startVal = abs((seed + blockId * 3 + threadId * 7 +
                      ((int)clock() / 10000000) % INT_MAX) %
                     INT_MAX);

  curandState_t state;
  curand_init(startVal, threadId + blockId, 11, &state);

  int inTolCount = 0;

  for (int iter = 0; iter < maxNumOfIters; iter++) {

    float normA = 0.0;
    float normB = 0.0;

    // a und b zufällig initialisieren
    for (int i = 0; i < nn; i++) {
      a[i] = 1.0 - (float)curand(&state) / (float)INT_MAX;
      b[i] = 1.0 - (float)curand(&state) / (float)INT_MAX;
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = 1.0 / sqrt(normA);
    normB = 1.0 / sqrt(normB);

    // normieren a und b:
    for (int i = 0; i < nn; i++) {
      a[i] *= normA;
      b[i] *= normB;
    }

    // korrektes c
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        c[i * n + j] = 0.0;
        for (int k = 0; k < n; k++) {
          c[i * n + j] += a[i * n + k] * b[k * n + j];
        }
      }
    }

    // c* = a* x b*
    for (int i = 0; i < p; i++) {
      aStar[i] = 0.0;
      bStar[i] = 0.0;
      for (int j = 0; j < nn; j++) {
        aStar[i] += myWa[nn * i + j] * a[j];
        bStar[i] += myWb[nn * i + j] * b[j];
      }
      cStar[i] = aStar[i] * bStar[i];
    }

    float err = 0.0;

    // c_wave - c  .. Fehler in c
    for (int i = 0; i < nn; i++) {
      float cWave = 0.0;
      for (int k = 0; k < p; k++) {
        cWave += myWc[p * i + k] * cStar[k];
      }
      cDiff[i] = cWave - c[i];
      err += cDiff[i] * cDiff[i];
    }

    err = sqrt(err);
    if (iter % (max(maxNumOfIters / 5, 100)) == 0 && iter > 0) {
      printf("kernel: block %i, thread %i, iter %i err = %f\n", blockId,
             threadId, iter, err);
    }

    if (isnan(err) || isinf(err) || err > 1000 || *killSignal == 1) {
      freeGb(garbageDump, garbageCounter);
      return;
    }

    if (err < tol) {
      inTolCount++;
      if (inTolCount > 10000) {
        lock(mutex);
        if (*killSignal == 1) {
          unlock(mutex);
          freeGb(garbageDump, garbageCounter);
          return;
        }

        float distance = 0.0;
        for (int i = 0; i < nn * p; i++) {
          distance += (Wa[i] - myWa[i]) * (Wa[i] - myWa[i]);
          distance += (Wb[i] - myWb[i]) * (Wb[i] - myWb[i]);
          distance += (Wc[i] - myWc[i]) * (Wc[i] - myWc[i]);
        }
        distance = sqrt(distance);

        printf("kernel: Solved by block %i, thread %i with err = %f distance = "
               "%f.\n",
               blockId, threadId, err, distance);

        (*distanceCount)++;
        if (*distanceCount >= minDistanceOutOf) {
          *killSignal = 1;
        }

        if (distance < *minDistance) {
          *minDistance = distance;
          for (int i = 0; i < nn * p; i++) {
            Wa[i] = myWa[i];
            Wb[i] = myWb[i];
            Wc[i] = myWc[i];
          }
        }

        unlock(mutex);
        freeGb(garbageDump, garbageCounter);
        return;
      }
    } else {
      inTolCount = 0;
    }

    // Korrektur Wa und Wb
    for (int i = 0; i < p; i++) {
      float WcTCDiff = 0.0;
      for (int j = 0; j < nn; j++) {
        WcTCDiff += myWc[i + j * p] * cDiff[j];
      }
      float WCBStar = WcTCDiff * bStar[i] * nueAB;
      float WCAStar = WcTCDiff * aStar[i] * nueAB;
      for (int j = 0; j < nn; j++) {
        if (useMasks) {
          myWa[i * nn + j] -= WCBStar * a[j] * Ma[i * nn + j];
          myWb[i * nn + j] -= WCAStar * b[j] * Mb[i * nn + j];
        } else {
          myWa[i * nn + j] -= WCBStar * a[j];
          myWb[i * nn + j] -= WCAStar * b[j];
        }
      }
    }

    // Korrektur Wc
    for (int i = 0; i < nn; i++) {
      float CDiffNue = cDiff[i] * nueC;
      for (int j = 0; j < p; j++) {
        if (useMasks) {
          myWc[i * p + j] -= CDiffNue * cStar[j] * Mc[i * p + j];
        } else {
          myWc[i * p + j] -= CDiffNue * cStar[j];
        }
      }
    }
  } // iter
} // kernel()

// memory operations and starting of kernels on GPU
float runBackpropOnGPU(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                       float *Mc, int maxNumIters, float nueAB, float nueC,
                       float tol, int n, int p, int seed, int blocks,
                       int threads, bool useMasks, int minDistanceOutOf) {

  std::cout << "runBackpropOnGPU: n = " << n << ", p = " << p << '\n';
  std::cout << "runBackpropOnGPU: blocks = " << blocks
            << ", threads = " << threads << '\n';
  std::cout << "runBackpropOnGPU: minDistanceOutOf = " << minDistanceOutOf
            << '\n';
  std::cout << "runBackpropOnGPU: masks on = " << useMasks << '\n';

  int nn = n * n;

  float *WaGPU, *WbGPU, *WcGPU;
  float *MaGPU, *MbGPU, *McGPU;
  float *minDistanceDevice;
  float minDistance = FLT_MAX;
  int *mutex, *killSignal, *distanceCount;

  size_t grantedMemSize;
  size_t demandedMemSize =
      (nn * p * 3 + nn * 4 + p * 3) * sizeof(float) * blocks * threads * 2;
  cudaDeviceGetLimit(&grantedMemSize, cudaLimitMallocHeapSize);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                     max(grantedMemSize, demandedMemSize));
  cudaDeviceGetLimit(&grantedMemSize, cudaLimitMallocHeapSize);
  std::cout << "runBackpropOnGPU: demandedMemSize = " << demandedMemSize
            << '\n';
  std::cout << "runBackpropOnGPU: grantedMemSize =  " << grantedMemSize << '\n';

  if (grantedMemSize < demandedMemSize)
    return -9.0; // mem. allocation declined

  cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

  checkForCudaError(217);

  cudaMalloc(&mutex, sizeof(int));
  cudaMemset(mutex, 0, sizeof(int));
  cudaMalloc(&killSignal, sizeof(int));
  cudaMemset(killSignal, 0, sizeof(int));
  cudaMalloc(&distanceCount, sizeof(int));
  cudaMemset(distanceCount, 0, sizeof(int));

  cudaMalloc(&minDistanceDevice, sizeof(float));
  cudaMalloc(&WaGPU, nn * p * sizeof(float));
  cudaMalloc(&WbGPU, nn * p * sizeof(float));
  cudaMalloc(&WcGPU, nn * p * sizeof(float));
  cudaMalloc(&MaGPU, nn * p * sizeof(float));
  cudaMalloc(&MbGPU, nn * p * sizeof(float));
  cudaMalloc(&McGPU, nn * p * sizeof(float));

  cudaMemcpy(minDistanceDevice, &minDistance, sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(WaGPU, Wa, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WbGPU, Wb, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WcGPU, Wc, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(MaGPU, Ma, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(MbGPU, Mb, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(McGPU, Mc, nn * p * sizeof(float), cudaMemcpyHostToDevice);

  checkForCudaError(244);

  dim3 blockGrid(blocks);
  dim3 threadGrid(threads);
  kernel<<<blockGrid, threadGrid>>>(WaGPU, WbGPU, WcGPU, MaGPU, MbGPU, McGPU,
                                    maxNumIters, nueAB, nueC, tol, n, p, seed,
                                    minDistanceDevice, mutex, killSignal,
                                    useMasks, minDistanceOutOf, distanceCount);

  checkForCudaError(252);

  cudaMemcpy(&minDistance, minDistanceDevice, sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(Wa, WaGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wb, WbGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wc, WcGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(killSignal);
  cudaFree(mutex);
  cudaFree(minDistanceDevice);
  cudaFree(WaGPU);
  cudaFree(WbGPU);
  cudaFree(WcGPU);
  cudaFree(MaGPU);
  cudaFree(MbGPU);
  cudaFree(McGPU);
  cudaThreadExit();

  checkForCudaError(272);

  std::cout << "runBackpropOnGPU: finished, minDistance: " << minDistance
            << '\n';
  return minDistance;
}
