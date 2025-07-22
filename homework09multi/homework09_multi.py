import scipy.io
import tensorly as tl
from tensorly.decomposition import parafac

# nome do arquivo .mat
filename = 'Practice_9_cpd_tensor.mat'

data = scipy.io.loadmat(filename)
tensor = data['tenX']

# parametros do algoritmo
R = 3 # nivel de rank 
delta = 1e-6 #criterio de convergencia
max_iter = 1000 #maximo de interacoes

# utilizando a funcao parafac da biblioteca tensorly
factors = parafac(tensor, rank=R, tol=delta, n_iter_max=max_iter)

# lista dos fatores estimados
A_hat = factors.factors
B_hat = factors.factors
C_hat = factors.factors

# print das matrizes estimadas
print("Matriz A estimada:\n", A_hat)
print("\nMatriz B estimada:\n", B_hat)
print("\nMatriz C estimada:\n", C_hat)

# 4. Calcular o erro
# Reconstruindo o tensor original a partir dos fatores
#tenX_reconstructed =
# Calculando o erro final (tl.norm ja calcula a norma de Frobenius por padrao para tensores)
#erro_final = tl.norm(tensor - tenX_reconstructed)

#print("Erro final (Norma de Frobenius da diferença):", erro_final)

