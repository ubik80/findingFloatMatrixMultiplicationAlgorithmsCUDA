
CUDA implementation of backpropagation for finding matrix multiplication algorithms with p products for n x n matrices. 

For further understanding of the subject look at: 
https://www.researchgate.net/publication/341942316_Searching_for_fast_matrix_multiplication_algorithms

This implementation finds the float solutions, for finding solutions in {-1, 0, 1}, have a look at:
https://github.com/ubik80/searching-for-fast-MM-algorithms

Starting from the same starting point, we iterate on many kernels until we find one solution. So, the aim is to find one solution not too far from the starting point. We could modify the solution for different goals, for example:

- let each kernel start at a different starting point (not usable in a projection algorithm).
- calculate several solutions, starting from the same starting point, and chooose the one with the smallest distanct to the starting point. This could be useful to calculate the solution with the smallest projection distance (see document mentioned above). 

The solution was developed and tested on:
- ubuntu 18.04.5
- python 3.6.9
- gcc 6.5.0
- RTX2070 GPU

pybind11 cloned from:
https://github.com/pybind/pybind11

use:
- clone this repo,
- clone the pybind11 repo
- do you need to install nvidia CUDA toolkit?
- cmake .
- cmake --build .
- set problem size / dims with: python3 setNAndP.py
- start with: python3 searchForMatrixMultiplicationAlgorithms



