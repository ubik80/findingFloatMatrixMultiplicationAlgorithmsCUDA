#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// implemented in the .cu file, calls kernels on GPU
float runBackpropOnGPU(float *Wa, float *Wb, float *Wc, int maxNumIters,
                       float nueAB, float nueC, float tol, int n, int p,
                       int seed, int blocks, int threads);

// to be called from python, through pybind11
template <typename T>
T backpropCUDA(py::array_t<T> _Wa, py::array_t<T> _Wb, py::array_t<T> _Wc,
               int maxNumOfIters, T nueAB, T nueC, T tol, int seed, int blocks,
               int threads) {

  auto bufWa = _Wa.request();
  auto bufWb = _Wb.request();
  auto bufWc = _Wc.request();
  int nn = bufWa.shape[1];
  int n = (int)sqrt(nn);
  int p = bufWa.shape[0];
  T *Wa = (T *)bufWa.ptr;
  T *Wb = (T *)bufWb.ptr;
  T *Wc = (T *)bufWc.ptr;
  float *Waf = (float *)malloc(nn * p * sizeof(float));
  float *Wbf = (float *)malloc(nn * p * sizeof(float));
  float *Wcf = (float *)malloc(nn * p * sizeof(float));

  // we want float on the GPU
  for (int i = 0; i < p * nn; i++) {
    Waf[i] = (float)Wa[i];
    Wbf[i] = (float)Wb[i];
    Wcf[i] = (float)Wc[i];
  }

  T ret =
      (T)runBackpropOnGPU(Waf, Wbf, Wcf, maxNumOfIters, (float)nueAB,
                          (float)nueC, (float)tol, n, p, seed, blocks, threads);

  for (int i = 0; i < p * nn; i++) {
    Wa[i] = (T)Waf[i];
    Wb[i] = (T)Wbf[i];
    Wc[i] = (T)Wcf[i];
  }

  free(Waf);
  free(Wbf);
  free(Wcf);

  return ret;
}

PYBIND11_PLUGIN(backpropCUDA) {
  py::module m("backprop on GPU", "backprop on GPU bybind11 plugin");
  m.def("backpropCUDA", backpropCUDA<double>);
  return m.ptr();
}
