import numpy as np


solution = np.load("solution_n3_p23.npy")

Wa = solution[0]
Wb = solution[1]
Wc = solution[2].T

input_A = np.array([
    [1, 2, 3],
    [-1, 2, 3],
    [3, -2, 1]], dtype=float)

input_B = np.array([
    [1, 2, -1],
    [3, 2, 3],
    [3, -1, 1]], dtype=float)

correct_solution = np.matmul(input_A, input_B)

flat_A = input_A.flatten()
flat_B = input_B.flatten()

WaA = np.matmul(Wa, flat_A)
WbB = np.matmul(Wb, flat_B)
C_star = np.multiply(WaA, WbB)

elser_C = np.matmul(Wc, C_star)

pass
