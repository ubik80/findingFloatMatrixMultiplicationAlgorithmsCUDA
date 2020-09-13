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

float runBackpropOnGPU(float *Wa, float *Wb, float *Wc, float *Ma, float *Mb,
                       float *Mc, int maxNumOfIters, float _nueAB, float _nueC,
                       float tol, int n, int p, int seed, int blocks,
                       int threads, bool useMasks, int minDistanceOutOf);

template <typename T>
T multipleBackpropMasked(py::array_t<T> _Wa, py::array_t<T> _Wb,
                         py::array_t<T> _Wc, py::array_t<T> _Ma,
                         py::array_t<T> _Mb, py::array_t<T> _Mc,
                         int maxNumOfIters, T nueAB, T nueC, T tol, int seed,
                         int blocks, int threads, bool useMasks,
                         int minDistanceOutOf) {

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
  T *Ma, *Mb, *Mc;
  if (useMasks) {
    Ma = (T *)bufMa.ptr;
    Mb = (T *)bufMb.ptr;
    Mc = (T *)bufMc.ptr;
  } else {
    Ma = nullptr;
    Mb = nullptr;
    Mc = nullptr;
  }
  float *Waf = (float *)malloc(nn * p * sizeof(float));
  float *Wbf = (float *)malloc(nn * p * sizeof(float));
  float *Wcf = (float *)malloc(nn * p * sizeof(float));

  float *Maf, *Mbf, *Mcf;
  if (useMasks) {
    float *Maf = (float *)malloc(nn * p * sizeof(float));
    float *Mbf = (float *)malloc(nn * p * sizeof(float));
    float *Mcf = (float *)malloc(nn * p * sizeof(float));
  } else {
    Maf = nullptr;
    Mbf = nullptr;
    Mcf = nullptr;
  }

  for (int i = 0; i < p * nn; i++) {
    Waf[i] = (float)Wa[i];
    Wbf[i] = (float)Wb[i];
    Wcf[i] = (float)Wc[i];
    if (useMasks) {
      Maf[i] = (float)Ma[i];
      Mbf[i] = (float)Mb[i];
      Mcf[i] = (float)Mc[i];
    }
  }

  T ret = (T)runBackpropOnGPU(Waf, Wbf, Wcf, Maf, Mbf, Mcf, maxNumOfIters,
                              (float)nueAB, (float)nueC, (float)tol, n, p, seed,
                              blocks, threads, useMasks, minDistanceOutOf);

  for (int i = 0; i < p * nn; i++) {
    Wa[i] = (T)Waf[i];
    Wb[i] = (T)Wbf[i];
    Wc[i] = (T)Wcf[i];
  }

  free(Waf);
  free(Wbf);
  free(Wcf);

  if (useMasks) {
    free(Maf);
    free(Mbf);
    free(Mcf);
  }

  return ret;
}

PYBIND11_PLUGIN(backpropCUDA) {
  py::module m("backprop on GPU", "backprop on GPU bybind11 plugin");
  m.def("multipleBackpropMasked", multipleBackpropMasked<double>);
  return m.ptr();
}
