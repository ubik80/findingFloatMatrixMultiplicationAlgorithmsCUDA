#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <sstream>

__global__ void kernel(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                       float *Mc, int maxNumOfIters, float _nueAB, float _nueC,
                       float tol, int n, int p, int seed) {

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
  float *a, *b, *c, *aStar, *bStar, *cStar, *cWave;
  a = (float *)malloc(matrixSizeABC);
  b = (float *)malloc(matrixSizeABC);
  c = (float *)malloc(matrixSizeABC);
  aStar = (float *)malloc(p);
  bStar = (float *)malloc(p);
  cStar = (float *)malloc(p);
  cWave = (float *)malloc(matrixSizeABC);

  curandState_t state;
  curand_init((seed + blockId * threadId) % INT_MAX, /* the seed controls the
                        sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for
                    each call, can be 0 */
              &state);

  // a und b zufällig initialisieren
  for (int i = 0; i < nn; i++) {
    a[i] = 1.0 - (float)curand(&state) / (float)INT_MAX;
    b[i] = 1.0 - (float)curand(&state) / (float)INT_MAX;
  }

  // korrektes c errechnen
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      c[i * n + j] = 0.0;
      for (int k = 0; k < n; k++) {
        c[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }

  // c* errechnen
  for (int i = 0; i < p; i++) {
    aStar[i] = 0.0;
    bStar[i] = 0.0;
    for (int j = 0; j < nn; j++) {
      aStar[i] += myWa[nn * i + j] * a[j];
      bStar[i] += myWb[nn * i + j] * a[j];
      cStar[i] = c[i] - aStar[i] * bStar[i];
    }
  }

  // "inkorrektes" c errechnen
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cWave[n * i + j] = 0.0;
      for (int k = 0; k < p; k++) {
        cWave[n * i + j] += myWc[p * i + k] * cStar[k];
      }
    }
  }
} // kernel

int runBackpropOnGPU(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                     float *Mc, int maxNumIters, float nueAB, float nueC,
                     float tol, int n, int p, int seed) {
  std::cout << "runBackpropOnGPU n = " << n << ", p = " << p << '\n';

  int nn = n * n;

  float *WaGPU, *WbGPU, *WcGPU;
  float *MaGPU, *MbGPU, *McGPU;

  cudaMalloc(&WaGPU, nn * p * sizeof(float));
  cudaMalloc(&WbGPU, nn * p * sizeof(float));
  cudaMalloc(&WcGPU, nn * p * sizeof(float));
  cudaMalloc(&MaGPU, nn * p * sizeof(float));
  cudaMalloc(&MbGPU, nn * p * sizeof(float));
  cudaMalloc(&McGPU, nn * p * sizeof(float));

  cudaMemcpy(WaGPU, Wa, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WbGPU, Wb, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(WcGPU, Wc, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(MaGPU, Ma, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(MbGPU, Mb, nn * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(McGPU, Mc, nn * p * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockGrid(36);
  dim3 threadGrid(36 * 32);
  kernel<<<blockGrid, threadGrid>>>(WaGPU, WbGPU, WcGPU, MaGPU, MbGPU, McGPU,
                                    maxNumIters, nueAB, nueC, tol, n, p, seed);

  cudaMemcpy(Wa, WaGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wb, WbGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Wc, WcGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ma, MaGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Mb, MbGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Mc, McGPU, nn * p * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(WaGPU);
  cudaFree(WbGPU);
  cudaFree(WcGPU);
  cudaFree(MaGPU);
  cudaFree(MbGPU);
  cudaFree(McGPU);
  cudaThreadExit();

  return 0;
}
