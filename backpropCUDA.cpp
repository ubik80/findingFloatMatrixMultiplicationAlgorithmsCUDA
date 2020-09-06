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

int runBackpropOnGPU(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                     float *Mc, int maxNumOfIters, float _nueAB, float _nueC,
                     float tol, int n, int p, int seed);

int multipleBackpropMasked(py::array_t<float> _Wa, py::array_t<float> _Wb,
                           py::array_t<float> _Wc, py::array_t<float> _Ma,
                           py::array_t<float> _Mb, py::array_t<float> _Mc,
                           int maxNumOfIters, float _nueAB, float _nueC,
                           float tol, int seed) {
  float nueAB = -_nueAB;
  float nueC = -_nueC;
  auto bufWa = _Wa.request();
  auto bufWb = _Wb.request();
  auto bufWc = _Wc.request();
  auto bufMa = _Ma.request();
  auto bufMb = _Mb.request();
  auto bufMc = _Mc.request();
  int nn = bufWa.shape[1];
  int n = (int)sqrt(nn);
  int p = bufWa.shape[0];
  float *Wa = (float *)bufWa.ptr;
  float *Wb = (float *)bufWb.ptr;
  float *Wc = (float *)bufWc.ptr;
  float *Ma = (float *)bufMa.ptr;
  float *Mb = (float *)bufMb.ptr;
  float *Mc = (float *)bufMc.ptr;

  return runBackpropOnGPU(Wa, Wb, Wc, Ma, Mb, Mc, maxNumOfIters, nueAB, nueC,
                          tol, n, p, seed);
}

PYBIND11_PLUGIN(backpropCUDA) {
  py::module m("example", "pybind example plugin");
  m.def("multipleBackpropMasked", multipleBackpropMasked);
  return m.ptr();
}
