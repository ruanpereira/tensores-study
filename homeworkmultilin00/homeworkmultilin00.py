import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# para o item a do primeiro problema
N = [2,4,8,16] # 32 e 64 nao foram testados devido ao tempo de execução

time_met1 = np.zeros(len(N))
time_met2 = np.zeros(len(N))

inter_max = 1000
for nn in range(len(N)):
    for i in tqdm(range(len(N))):
        time.sleep(0.01)
        current_n = N[nn]
        for i in range(inter_max):
            A = np.random.randn(current_n, current_n) + 1j * np.random.randn(current_n, current_n)
            B = np.random.randn(current_n, current_n) + 1j * np.random.randn(current_n, current_n)
            # para o metodo 1
            start_time = time.time()
            kron_prod = np.kron(A, B)
            np.linalg.inv(kron_prod)
            end_time = time.time()
            time_met1[nn] += end_time - start_time
            print(f"tempo do metodo 1: {time_met1[nn]}")
            # para o metodo 2
            start_time = time.time()
            inv_A = np.linalg.inv(A)
            inv_B = np.linalg.inv(B)
            np.kron(inv_A, inv_B)
            end_time = time.time()
            time_met2[nn] += end_time - start_time
            print(f"tempo do metodo 2: {time_met2[nn]}")

# calculando o tempo médio
time_met1 /= inter_max
time_met2 /= inter_max

print(f"tempo médio do metodo 1: {time_met1}")
print(f"tempo médio do metodo 2: {time_met2}")

plt.figure(figsize=(10, 6))
plt.plot(N, time_met1, 'o-', label='Método 1: inv(kron(A, B))')
plt.plot(N, time_met2, 's-', label='Método 2: kron(inv(A), inv(B))')
plt.xlabel("Tamanho da Matriz (N)")
plt.ylabel("Tempo Médio de Execução (s)")
plt.title("Comparação de Desempenho Computacional item A")
plt.legend()
plt.yscale('log')
plt.xticks(N)
plt.show()

# para o item b do primeiro problema
K = [2, 4, 6] # 8 e 10 nao foram testados devido ao tempo de execução
N = 4

timeb_met1 = np.zeros(len(K))
timeb_met2 = np.zeros(len(K))

inter_max = 1000
for kk in range(len(K)):
    for i in tqdm(range(len(K))):
        time.sleep(0.01)
        current_k = K[kk]
        for i in range(inter_max):
                A = [np.random.randn(N,N) + 1j * np.random.randn(N,N) for _ in range(current_k)]
                # para o metodo 1
                start_time = time.time()
                kron_prod = A[0]
                if current_k > 1:
                    for i in range(1, current_k):
                        kron_prod = np.kron(kron_prod, A[i])
                np.linalg.inv(kron_prod)
                end_time = time.time()
                timeb_met1[kk] += end_time - start_time
                #print(f"tempo do metodo 1: {timeb_met1[kk]}")
                # para o metodo 2
                start_time = time.time()
                inv_A = [np.linalg.inv(A[i]) for _ in range(current_k)]
                kron_prod_inv = inv_A[0]
                if current_k > 1:
                    for i in range(1, current_k):
                        kron_prod = np.kron(kron_prod_inv, inv_A[i])
                end_time = time.time()
                timeb_met2[kk] += end_time - start_time
                #print(f"tempo do metodo 2: {timeb_met2[kk]}")

# Calcula o tempo médio
timeb_met1 /= inter_max
timeb_met2 /= inter_max

print(f"tempo médio do metodo 1 item b: {timeb_met1}")
print(f"tempo médio do metodo 2 item b: {timeb_met2}")

plt.figure(figsize=(10,6))
plt.plot(K, timeb_met1, 'o-', label='Método 1: inv(kron(A^K)')
plt.plot(K, timeb_met2, 's-', label='Método 2: kron(inv(A^K)')
plt.xlabel("Número de Matrizes (K)")
plt.ylabel("Tempo Médio de Execução (s)")
plt.title("Comparação de Desempenho Computacional - Item B")
plt.legend()
plt.yscale('log')
plt.xticks(K)
plt.show()