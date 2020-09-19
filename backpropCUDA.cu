#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>

// memory allocation, remember pointers for cleanup later
__device__ float *mallocGb(int numOfFloats, float **garbageDump,
                           int garbageCounter) {
  garbageDump[garbageCounter] = (float *)malloc(numOfFloats * sizeof(float));
  garbageCounter++;
  return garbageDump[garbageCounter - 1];
}

// free all allocated (pointers in garbageDump)
__device__ void freeGb(float **garbageDump, int garbageCounter) {
  for (int i = 0; i < garbageCounter; i++) {
    free(garbageDump[i]);
  }
}

// to protect Wa, Wb, Wc and minError
__device__ void lock(int *mutex) {
  while (atomicCAS(mutex, 0, 1) != 0) {
  };
}

// to protect Wa, Wb, Wc and minError
__device__ void unlock(int *mutex) { atomicExch(mutex, 0); }

// randomly initialize Wa, Wb, Wc
__device__ void initializeWaWbWc(float *Wa, float *Wb, float *Wc,
                                 curandState_t *state, int nn, int p) {
  for (int i = 0; i < nn * p; i++) {
    Wa[i] = 1.0 - (float)curand(state) / (float)INT_MAX;
    Wb[i] = 1.0 - (float)curand(state) / (float)INT_MAX;
    Wc[i] = 1.0 - (float)curand(state) / (float)INT_MAX;
  }
}

// randomly set a and b, and scale to length 1
__device__ void initializeAB(float *a, float *b, curandState_t *state, int nn,
                             int p) {
  float normA = 0.0;
  float normB = 0.0;

  do {
    for (int i = 0; i < nn; i++) {
      a[i] = 1.0 - (float)curand(state) / (float)INT_MAX;
      b[i] = 1.0 - (float)curand(state) / (float)INT_MAX;
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
  } while (normA < 0.1 or normB < 0.1);

  normA = 1.0 / sqrt(normA);
  normB = 1.0 / sqrt(normB);

  for (int i = 0; i < nn; i++) {
    a[i] *= normA;
    b[i] *= normB;
  }
}

// calculate c (mat(c)=mat(a)*mat(b))
__device__ void calculateC(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      c[i * n + j] = 0.0;
      for (int k = 0; k < n; k++) {
        c[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
}

// c* = a* x b*
__device__ void calculateCStar(float *Wa, float *Wb, float *aStar, float *bStar,
                               float *cStar, float *a, float *b, int nn,
                               int p) {
  for (int i = 0; i < p; i++) {
    aStar[i] = 0.0;
    bStar[i] = 0.0;
    for (int j = 0; j < nn; j++) {
      aStar[i] += Wa[nn * i + j] * a[j];
      bStar[i] += Wb[nn * i + j] * b[j];
    }
    cStar[i] = aStar[i] * bStar[i];
  }
}

__device__ float calculateCDiffAndErr(float *c, float *cStar, float *Wc,
                                      float *cDiff, int nn, int p) {
  float cWave;
  float err = 0.0;
  for (int i = 0; i < nn; i++) {
    cWave = 0.0;
    for (int k = 0; k < p; k++) {
      cWave += Wc[p * i + k] * cStar[k];
    }
    cDiff[i] = cWave - c[i]; // c_wave - c is the error in c
    err += cDiff[i] * cDiff[i];
  }
  return sqrt(err);
}

// innovate Wa and Wb
__device__ void innovateWaAndWb(float *aStar, float *bStar, float *a, float *b,
                                float *cDiff, float *Wa, float *Wb, float *Wc,
                                float nueAB, int nn, int p) {
  float WcTCDiff, WCAStar, WCBStar;
  for (int i = 0; i < p; i++) {
    WcTCDiff = 0.0;
    for (int j = 0; j < nn; j++) {
      WcTCDiff += Wc[i + j * p] * cDiff[j];
    }
    WCBStar = WcTCDiff * bStar[i] * nueAB;
    WCAStar = WcTCDiff * aStar[i] * nueAB;
    for (int j = 0; j < nn; j++) {
      Wa[i * nn + j] -= WCBStar * a[j];
      Wb[i * nn + j] -= WCAStar * b[j];
    }
  }
}

// innovate Wc
__device__ void innovateWc(float *cStar, float *cDiff, float *Wc, float nueC,
                           int nn, int p) {
  float CDiffNue;
  for (int i = 0; i < nn; i++) {
    CDiffNue = cDiff[i] * nueC;
    for (int j = 0; j < p; j++) {
      Wc[i * p + j] -= CDiffNue * cStar[j];
    }
  }
}

__global__ void kernel(float *Wa, float *Wb, float *Wc, int maxNumOfIters,
                       float nueAB, float nueC, float tol, int n, int p,
                       int seed, int *killSignal, int *mutex, float *minError) {
  float *garbageDump[10]; // for cleanup before return
  int garbageCounter = 0;

  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;
  const int nn = n * n;

  float *myWa = (float *)mallocGb(nn * p, garbageDump, garbageCounter);
  float *myWb = (float *)mallocGb(nn * p, garbageDump, garbageCounter);
  float *myWc = (float *)mallocGb(nn * p, garbageDump, garbageCounter);
  float *a = (float *)mallocGb(nn, garbageDump, garbageCounter);
  float *b = (float *)mallocGb(nn, garbageDump, garbageCounter);
  float *c = (float *)mallocGb(nn, garbageDump, garbageCounter);
  float *aStar = (float *)mallocGb(p, garbageDump, garbageCounter);
  float *bStar = (float *)mallocGb(p, garbageDump, garbageCounter);
  float *cStar = (float *)mallocGb(p, garbageDump, garbageCounter);
  float *cDiff = (float *)mallocGb(nn, garbageDump, garbageCounter);

  int startVal =
      abs((seed + blockId * 3 + threadId * 7 + blockId * threadId * 11 +
           ((int)clock() / 10000000) % INT_MAX) %
          INT_MAX);
  curandState_t state;
  curand_init(startVal, threadId + blockId, 13, &state);

  initializeWaWbWc(myWa, myWb, myWc, &state, nn, p);

  int inTolCount = 0;                // counts iterations with err < tol
  int printCount = 0;                // for cmdl output
  int printFreq = maxNumOfIters / 5; // how often to print
  float err;

  for (int iter = 0; iter < maxNumOfIters; iter++) {

    if (*killSignal > 0) {
      freeGb(garbageDump, garbageCounter);
      return;
    }

    initializeAB(a, b, &state, nn, p);

    calculateC(a, b, c, n);

    calculateCStar(myWa, myWb, aStar, bStar, cStar, a, b, nn, p);

    err = calculateCDiffAndErr(c, cStar, myWc, cDiff, nn, p);

    if (printCount == printFreq) {
      printCount = 0;
      printf("kernel: block %i, thread %i, iter %i err = %f\n", blockId,
             threadId, iter, err);
    }
    printCount++;

    if (isnan(err) || isinf(err) || isinf(-err) || err > 10000) {
      initializeWaWbWc(myWa, myWb, myWc, &state, nn, p); // Wa, Wb, Wc corrupted
    }

    innovateWaAndWb(aStar, bStar, a, b, cDiff, myWa, myWb, myWc, nueAB, nn, p);

    innovateWc(cStar, cDiff, myWc, nueC, nn, p);

    if (err < tol) {
      inTolCount++;
      if (inTolCount > 10000) { // 10000 cycles with err < tol --> finished
        lock(mutex);
        if (*killSignal > 0) {
          unlock(mutex);
          freeGb(garbageDump, garbageCounter);
          return;
        }
        atomicAdd(killSignal, 1);
        printf("kernel: Solved by block %i, thread %i with err = %f.\n ",
               blockId, threadId, err);
        for (int i = 0; i < nn * p; i++) { // write back result
          Wa[i] = myWa[i];
          Wb[i] = myWb[i];
          Wc[i] = myWc[i];
        }
        *minError = err;
        unlock(mutex);
        freeGb(garbageDump, garbageCounter);
        return;
      }
    } else { // err > tol
      inTolCount = 0;
    }
  } // iter
} // kernel()

// memory operations and starting of kernels on GPU
float runBackpropOnGPU(float *Wa, float *Wb, float *Wc, int maxNumIters,
                       float nueAB, float nueC, float tol, int n, int p,
                       int seed, int blocks, int threads) {

  std::cout << "runBackpropOnGPU: n = " << n << ", p = " << p << '\n';
  std::cout << "runBackpropOnGPU: blocks = " << blocks
            << ", threads = " << threads << '\n';

  int nn = n * n;

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
    return FLT_MAX; // mem. allocation declined

  cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);

  float minError = FLT_MAX;
  float *WaGPU, *WbGPU, *WcGPU, *minErrorGPU;
  int *killSignal, *mutex;

  cudaMalloc(&killSignal, sizeof(int));
  cudaMemset(killSignal, 0, sizeof(int));
  cudaMalloc(&mutex, sizeof(int));
  cudaMemset(mutex, 0, sizeof(int));
  cudaMalloc(&minErrorGPU, sizeof(float));
  cudaMemcpy(minErrorGPU, &minError, sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&WaGPU, nn * p * sizeof(float));
  cudaMalloc(&WbGPU, nn * p * sizeof(float));
  cudaMalloc(&WcGPU, nn * p * sizeof(float));

  dim3 blockGrid(blocks);
  dim3 threadGrid(threads);
  kernel<<<blockGrid, threadGrid>>>(WaGPU, WbGPU, WcGPU, maxNumIters, nueAB,
                                    nueC, tol, n, p, seed, killSignal, mutex,
                                    minErrorGPU);

  cudaMemcpy(&minError, minErrorGPU, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wa, WaGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wb, WbGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wc, WcGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(killSignal);
  cudaFree(mutex);
  cudaFree(minErrorGPU);
  cudaFree(WaGPU);
  cudaFree(WbGPU);
  cudaFree(WcGPU);

  cudaThreadExit();

  std::cout << "runBackpropOnGPU: finished" << '\n';
  return minError;
}
