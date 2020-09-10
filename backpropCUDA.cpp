#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>

namespace py = pybind11;

float runBackpropOnGPU( float *Wa, float *Wb, float *Wc,
                        float *Ma, float *Mb, float *Mc,
                        int maxNumOfIters,
                        float _nueAB, float _nueC, float tol,
                        int n, int p, int seed,
                        int blocks, int threads);

template <typename T>
T multipleBackpropMasked(py::array_t<T> _Wa, py::array_t<T> _Wb,
                         py::array_t<T> _Wc, py::array_t<T> _Ma,
                         py::array_t<T> _Mb, py::array_t<T> _Mc,
                         int maxNumOfIters, T nueAB, T nueC, T tol, int seed,
                         int blocks, int threads) {

  auto bufWa = _Wa.request();
  auto bufWb = _Wb.request();
  auto bufWc = _Wc.request();
  auto bufMa = _Ma.request();
  auto bufMb = _Mb.request();
  auto bufMc = _Mc.request();
  int nn = bufWa.shape[1];
  int n = (int)sqrt(nn);
  int p = bufWa.shape[0];
  T *Wa = (T *)bufWa.ptr;
  T *Wb = (T *)bufWb.ptr;
  T *Wc = (T *)bufWc.ptr;
  T *Ma = (T *)bufMa.ptr;
  T *Mb = (T *)bufMb.ptr;
  T *Mc = (T *)bufMc.ptr;
  float *Waf = (float *)malloc(nn * p * sizeof(float));
  float *Wbf = (float *)malloc(nn * p * sizeof(float));
  float *Wcf = (float *)malloc(nn * p * sizeof(float));
  float *Maf = (float *)malloc(nn * p * sizeof(float));
  float *Mbf = (float *)malloc(nn * p * sizeof(float));
  float *Mcf = (float *)malloc(nn * p * sizeof(float));

  for (int i = 0; i < p * nn; i++) {
    Waf[i] = (float)Wa[i];
    Wbf[i] = (float)Wb[i];
    Wcf[i] = (float)Wc[i];
    Maf[i] = (float)Ma[i];
    Mbf[i] = (float)Mb[i];
    Mcf[i] = (float)Mc[i];
  }

  return (T)runBackpropOnGPU(Waf, Wbf, Wcf, Maf, Mbf, Mcf, maxNumOfIters,
                          (float)nueAB, (float)nueC, (float)tol, n, p, seed,
                          blocks, threads);
}

PYBIND11_PLUGIN(backpropCUDA) {
  py::module m("backprop on GPU", "backprop on GPU bybind11 plugin");
  m.def("multipleBackpropMasked", multipleBackpropMasked<double>);
  return m.ptr();
}
