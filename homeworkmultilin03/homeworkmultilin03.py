import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import svd

def khatri_rao(A, B):
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrizes devem ter o mesmo número de colunas.")
    
    C = np.zeros((A.shape[0] * B.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        C[:, i] = np.kron(A[:, i], B[:, i])
    return C

def unvec(vec, nrow, ncol):
    return vec.reshape(nrow, ncol, order='F')

A = np.random.randn(4,2) + 1j * np.random.randn(4,2)
B = np.random.randn(6,2) + 1j * np.random.randn(6,2)

X = khatri_rao(A, B)

unvec_X0 = unvec(X[:, 0], 4, 6)
unvec_X1 = unvec(X[:, 1], 4, 6)
print(f'unvec_X primeira col: {unvec_X0}')
print(f'unvec_X segunda col: {unvec_X1}')

U0, S0, V0 = svd(unvec_X0)
U1, S1, V1 = svd(unvec_X1)
print(f'SVD primeira col: {U0}, {S0}, {V0}')
print(f'SVD segunda col: {U1}, {S1}, {V1}')

a_estm = np.sqrt(S0[0]) * np.conj(V0[:, 1])
print(f'temos entao a matriz A: {A[:, 0]}, e a matriz A estimada: {a_estm}')
