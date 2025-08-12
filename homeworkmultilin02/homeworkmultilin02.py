import numpy as np
import time
import matplotlib.pyplot as plt

def khatri_rao(A, B):
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matrizes devem ter o mesmo número de colunas.")
    
    C = np.zeros((A.shape[0] * B.shape[0], A.shape[1]))
    for i in range(A.shape[1]):
        C[:, i] = np.kron(A[:, i], B[:, i])
    return C

# para o problema 1
I_values = [2, 4, 8, 16, 32, 64] # 128, 256 foram retirados devido ao tempo de execução
R = 4

times1 = []
times2 = []
times3 = []

for I in I_values:
    print(f"Testing for I = {I}, R = {R}")
    
    A = np.random.randn(I, R) + 1j * np.random.randn(I, R)
    B = np.random.randn(I, R)
    
    X = khatri_rao(A, B)
    
    #metodo 1: pinv
    start_time = time.time()
    pinv_X1 = np.linalg.pinv(X)
    end_time = time.time()
    times1.append(end_time - start_time)
    
    # metodo 2: (X^H * X)^-1 * X^H
    start_time = time.time()
    X_H = X.conj().T
    XHX = X_H @ X
    XHX_inv = np.linalg.inv(XHX)
    pinv_X2 = XHX_inv @ X_H
    end_time = time.time()
    times2.append(end_time - start_time)

    # metodo 3: (A^H A) * (B^H B)
    start_time = time.time()
    A_H = A.conj().T
    B_T = B.T
    AHA = A_H @ A
    BTB = B_T @ B
    XHX_prop = AHA * BTB
    XHX_prop_inv = np.linalg.inv(XHX_prop)
    pinv_X3 = XHX_prop_inv @ X_H # usando X_H do metodo 2
    end_time = time.time()
    times3.append(end_time - start_time)

# plot
plt.figure(figsize=(12, 8))
plt.loglog(I_values, times1, 'o-', label='Method 1: pinv(X)')
plt.loglog(I_values, times2, 's-', label=r'Method 2: $(X^H X)^{-1} X^H$')
plt.loglog(I_values, times3, '^-', label='Method 3: Property-based')
plt.title('Comparativos entre os 3 métodos para R=4')
plt.xlabel('Number of Rows (I)')
plt.ylabel('Runtime (seconds)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

# para o problema 2
I = 4
R = 2
N = [2,4,6,8,10]
max_inter = 50

tempo = []

for n in N:
    total_tempo = 0
    for _ in range(max_inter):
        # gerando as matrizes aleatorias
        A = [np.random.rand(I, R) for _ in range(n)]
        start_time = time.time()
        X = A[0]
        for m in range(1,n):
            X = khatri_rao(X, A[m])
        end_time = time.time()
        total_tempo += end_time - start_time
    tempo.append(total_tempo / max_inter)

plt.figure(figsize=(10, 6))
# A semi-log plot is best to visualize exponential growth
plt.semilogy(N, tempo, 'o-', color='purple')

plt.title('Khatri-Rao Product Runtime vs. Number of Matrices (N)')
plt.xlabel('Number of Matrices (N)')
plt.ylabel('Average Runtime (seconds, log scale)')
plt.grid(True, which="both", ls="--")
plt.xticks(N)
plt.show()

