#Cuda implementation of backpropagation for finding matrix-multiplication-algorithms with p products for n x n matrices

Basically, n (matrix size) and p (number of products to use in MM-Algorithm) are set. Then 'python3 searchForMatrixMultiplicationAlgorithms' is started. If you are lucky, it will return the MM-Algorithm.

For further understanding of the subject look at: 
https://www.researchgate.net/publication/341942316_Searching_for_fast_matrix_multiplication_algorithms

This implementation finds the float solutions. For finding solutions in {-1, 0, 1}, have a look at:
https://github.com/ubik80/searching-for-fast-MM-algorithms

Strategy:
With different (random) starting points, we iterate on many kernels until we find one solution.

The solution was developed and tested on:
- ubuntu 22.04
- python 3.10.12
- gcc 11.4.0
- RTX4070 GPU

pybind11 cloned from:
https://github.com/pybind/pybind11

use:
- clone this repo,
- clone the pybind11 repo
- is your GPU driver suitable?
- do you need to install nvidia CUDA toolkit? Get it on the nvidia website.
- 'cmake .'
- 'cmake --build .'
- set problem size / dims with: 'python3 setNAndP.py'
- start with: 'python3 searchForMatrixMultiplicationAlgorithms'
- check found solution with 'python3 checkSolution.py'



