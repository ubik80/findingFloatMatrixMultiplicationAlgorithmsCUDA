#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

void checkForCudaError(int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
    fprintf(stderr, "cudaCheckError() failed in line %i:\t%s\n", line,
            cudaGetErrorString(err));
}

__device__ void lock(int *mutex){
  while (atomicCAS(mutex,0,1)!=0);
}

__device__ void unlock(int* mutex){
  atomicExch(mutex,0);
}

__global__ void kernel(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                       float *Mc, int maxNumOfIters, float nueAB, float nueC,
                       float tol, int n, int p, int seed, float *finalError,
                       int *mutex, int* killSignal) {

  const int threadId = threadIdx.x;
  const int blockId = blockIdx.x;

  const int nn = n * n;
  const int matrixSizeW = nn * p * sizeof(float);
  const int matrixSizeABC = nn * sizeof(float);
  float *myWa, *myWb, *myWc;
  myWa = (float *)malloc(matrixSizeW);
  myWb = (float *)malloc(matrixSizeW);
  myWc = (float *)malloc(matrixSizeW);
  memcpy(myWa, Wa, matrixSizeW);
  memcpy(myWb, Wb, matrixSizeW);
  memcpy(myWc, Wc, matrixSizeW);
  float *a, *b, *c, *aStar, *bStar, *cStar, *cDiff;
  a = (float *)malloc(matrixSizeABC);
  b = (float *)malloc(matrixSizeABC);
  c = (float *)malloc(matrixSizeABC);
  aStar = (float *)malloc(p);
  bStar = (float *)malloc(p);
  cStar = (float *)malloc(p);
  cDiff = (float *)malloc(matrixSizeABC);

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
    if (iter % (maxNumOfIters / 10) == 0 && iter > 0) {
      printf("block %i, thread %i, iter %i err = %f\n", blockId, threadId, iter,
             err);
    }

    if (isnan(err) || isinf(err) || err > 1000 || *killSignal == 1) {
      return;
    }

    if (err < tol) {
      inTolCount++;
      if (inTolCount > 100) {
        lock(mutex);
        if(*killSignal == 1){
          unlock(mutex);
          return;
        }
        if(err < *finalError){
          *finalError = err;
          *killSignal = 1;

          for(int i = 0; i < nn*p; i++){
            Wa[i] = myWa[i];
            Wb[i] = myWb[i];
            Wc[i] = myWc[i];
          }

          printf("beendet durch block %i, thread %i mit err = %f \n", blockId,
                  threadId, err);
          unlock(mutex);
          return;
        }
        unlock(mutex);
        return;
      }
    }
    else {
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
        myWa[i * nn + j] -= WCBStar * a[j] * Ma[i * nn + j];
        myWb[i * nn + j] -= WCAStar * b[j] * Mb[i * nn + j];
      }
    }

    // Korrektur Wc
    for (int i = 0; i < nn; i++) {
      float CDiffNue = cDiff[i] * nueC;
      for (int j = 0; j < p; j++) {
        myWc[i * p + j] -= CDiffNue * cStar[j] * Ma[i * p + j];
      }
    }
  } // iter
} // kernel

float runBackpropOnGPU(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                       float *Mc, int maxNumIters, float nueAB, float nueC,
                       float tol, int n, int p, int seed, int blocks, int threads) {
  std::cout << "runBackpropOnGPU n = " << n << ", p = " << p << '\n';

  int nn = n * n;

  float *WaGPU, *WbGPU, *WcGPU;
  float *MaGPU, *MbGPU, *McGPU;
  float *finalErrorDevice;
  float finalError = tol + 1.0;
  int *mutex, *killSignal;

  cudaMalloc(&mutex, sizeof(int));
  cudaMemset(mutex, 0, sizeof(int));
  cudaMalloc(&killSignal, sizeof(int));
  cudaMemset(killSignal, 0, sizeof(int));

  cudaMalloc(&finalErrorDevice, sizeof(float));
  cudaMalloc(&WaGPU, nn * p * sizeof(float));
  cudaMalloc(&WbGPU, nn * p * sizeof(float));
  cudaMalloc(&WcGPU, nn * p * sizeof(float));
  cudaMalloc(&MaGPU, nn * p * sizeof(float));
  cudaMalloc(&MbGPU, nn * p * sizeof(float));
  cudaMalloc(&McGPU, nn * p * sizeof(float));

  cudaMemcpy(finalErrorDevice, &finalError, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WaGPU, Wa, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WbGPU, Wb, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WcGPU, Wc, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(MaGPU, Ma, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(MbGPU, Mb, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(McGPU, Mc, nn * p * sizeof(float), cudaMemcpyHostToDevice);

  checkForCudaError(186);

  dim3 blockGrid(blocks);
  dim3 threadGrid(threads);
  kernel<<<blockGrid, threadGrid>>>(WaGPU, WbGPU, WcGPU, MaGPU, MbGPU, McGPU,
                                    maxNumIters, nueAB, nueC, tol, n, p, seed,
                                    finalErrorDevice, mutex, killSignal);

  checkForCudaError(194);

  cudaMemcpy(&finalError, finalErrorDevice, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wa, WaGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wb, WbGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wc, WcGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ma, MaGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Mb, MbGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Mc, McGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(killSignal);
  cudaFree(mutex);
  cudaFree(finalErrorDevice);
  cudaFree(WaGPU);
  cudaFree(WbGPU);
  cudaFree(WcGPU);
  cudaFree(MaGPU);
  cudaFree(MbGPU);
  cudaFree(McGPU);
  cudaThreadExit();

  checkForCudaError(215);

  return finalError;
}
