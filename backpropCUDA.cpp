#include <cuda_runtime.h>
#include <iostream>
#include "./pybind11/include/pybind11/numpy.h"
#include "./pybind11/include/pybind11/pybind11.h"
#include "./pybind11/include/pybind11/stl.h"
#include <float.h>

namespace py = pybind11;


// implemented in the .cu file, calls kernels on GPU
float runBackpropOnGPU(float *Wa, float *Wb, float *Wc, int maxNumIters,
                       float nueAB, float nueC, float tol, int n, int p,
                       int seed, int blocks, int threads);


// to be called from python, through pybind11
float backpropCUDA(py::array_t<float> _Wa, py::array_t<float> _Wb, py::array_t<float> _Wc,
               int maxNumOfIters, float nueAB, float nueC, float tol, int seed, int blocks,
               int threads) {

  py::buffer_info bufWa = _Wa.request();
  py::buffer_info bufWb = _Wb.request();
  py::buffer_info bufWc = _Wc.request();
  int nn = bufWa.shape[1];
  int n = (int)sqrt(nn);
  int p = bufWa.shape[0];
  float *Wa = (float *)bufWa.ptr;
  float *Wb = (float *)bufWb.ptr;
  float *Wc = (float *)bufWc.ptr;
  float *Waf = (float *)malloc(nn * p * sizeof(float));
  float *Wbf = (float *)malloc(nn * p * sizeof(float));
  float *Wcf = (float *)malloc(nn * p * sizeof(float));

  float ret = runBackpropOnGPU(Waf, Wbf, Wcf, maxNumOfIters, nueAB, nueC, tol, n, p, seed, blocks, threads);

  for (int i = 0; i < p * nn; i++) {
    Wa[i] = Waf[i];
    Wb[i] = Wbf[i];
    Wc[i] = Wcf[i];
  }

  free(Waf);
  free(Wbf);
  free(Wcf);

  return ret;
}


PYBIND11_MODULE(backpropCUDA, m) {
  m.def("backpropCUDA", &backpropCUDA);
}
