import numpy as np
import timeit
import matplotlib.pyplot as plt

# Problema 1: Produto de Hadamard
def hadamard_product(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrizes devem ter as mesmas dimensões para o produto de Hadamard")
    return np.multiply(A, B)  

# Problema 2: Produto de Kronecker
def kronecker_product(A, B):
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m*p, n*q), dtype=A.dtype)
    
    for i in range(m):
        for j in range(n):
            result[i*p:(i+1)*p, j*q:(j+1)*q] = A[i,j] * B
    return result

# Problema 3: Produto de Khatri-Rao
def kr(A, B):
    m, n = A.shape
    p, q = B.shape
    
    if n != q:
        raise ValueError("Para o produto Khatri-Rao, o número de colunas deve ser igual")
    
    result = np.zeros((m*p, n), dtype=A.dtype)
    
    for j in range(n):
        result[:, j] = np.kron(A[:, j], B[:, j])
    
    return result

# Função para medir tempos de execução
def benchmark():
    sizes = [2, 4, 8, 16, 32] #64 e 128 removidos por conta do tempo de compilação
    hadamard_times_manual = []
    hadamard_times_np = []
    kronecker_times_manual = []
    kronecker_times_np = []
    khatri_rao_times_manual = []
    khatri_rao_times_np = []
    
    for N in sizes:
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        B = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        
        # Benchmark Hadamard
        t = timeit.timeit(lambda: hadamard_product(A, B), number=10)
        hadamard_times_manual.append(t)
        
        t = timeit.timeit(lambda: A * B, number=10)
        hadamard_times_np.append(t)
        
        # Benchmark Kronecker
        t = timeit.timeit(lambda: kronecker_product(A, B), number=5)
        kronecker_times_manual.append(t)
        
        t = timeit.timeit(lambda: np.kron(A, B), number=5)
        kronecker_times_np.append(t)
        
        # Benchmark Khatri-Rao (assumindo N colunas)
        t = timeit.timeit(lambda: kr(A, B), number=5)
        khatri_rao_times_manual.append(t)
        
        # Não há função direta no NumPy para Khatri-Rao, usaremos uma implementação baseada em Kronecker
        t = timeit.timeit(lambda: np.concatenate([np.kron(A[:,j:j+1], B[:,j:j+1]) for j in range(N)], axis=1), number=5)
        khatri_rao_times_np.append(t)
    
    plt.figure(figsize=(15, 5))
    
    # Hadamard
    plt.subplot(1, 3, 1)
    plt.plot(sizes, hadamard_times_manual, label='Manual')
    plt.plot(sizes, hadamard_times_np, label='NumPy')
    plt.title('Produto de Hadamard')
    plt.xlabel('Tamanho da matriz (N)')
    plt.ylabel('Tempo (s)')
    plt.legend()
    
    # Kronecker
    plt.subplot(1, 3, 2)
    plt.plot(sizes, kronecker_times_manual, label='Manual')
    plt.plot(sizes, kronecker_times_np, label='NumPy (kron)')
    plt.title('Produto de Kronecker')
    plt.xlabel('Tamanho da matriz (N)')
    plt.ylabel('Tempo (s)')
    plt.legend()
    
    # Khatri-Rao
    plt.subplot(1, 3, 3)
    plt.plot(sizes, khatri_rao_times_manual, label='Manual')
    plt.plot(sizes, khatri_rao_times_np, label='NumPy (baseado em kron)')
    plt.title('Produto de Khatri-Rao')
    plt.xlabel('Tamanho da matriz (N)')
    plt.ylabel('Tempo (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Executar benchmark
if __name__ == "__main__":
    benchmark()