import scipy.io
import numpy as np

# Nome do arquivo .mat
filename = 'Homework_02_data.mat'

# Carrega os dados do arquivo .mat
mat_data = scipy.io.loadmat(filename)

print("Tipo do objeto carregado:", type(mat_data))
print("Conteúdo do arquivo (chaves do dicionário):", mat_data.keys())

# Acessando as variáveis como em um dicionário
# As variáveis são carregadas como arrays NumPy
matriz_A = mat_data['A']
escalar_b = mat_data['b']

print("\n--- Matriz A ---")
print(matriz_A)
print("Tipo da Matriz A:", type(matriz_A))


print("\n--- Escalar B ---")
print(escalar_b)
# Note que mesmo escalares são encapsulados em arrays
print("Valor do escalar B:", escalar_b[0, 0])