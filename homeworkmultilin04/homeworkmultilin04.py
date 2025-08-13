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

def kron(list_of_arrays):
    shapes = []
    for i in range(0, len(list_of_arrays)):
        shape_of_array = list_of_arrays[i].shape
        shapes.append(shape_of_array)

    matrix = list_of_arrays[0]
    for i in range(1, len(list_of_arrays)):
        matrix = matrix[:, np.newaxis, :, np.newaxis] * list_of_arrays[i][np.newaxis, :, np.newaxis, :]

        shapes_aux = np.ones([np.size(shapes[0])], dtype='int8')
        for j in range(i - 1, i + 1):
            for k in range(0, np.size(shapes[0])):
                shapes_aux[k] = shapes_aux[k] * shapes[j][k]

        matrix = matrix.reshape(shapes_aux)
        shapes[i] = matrix.shape

    return matrix

def unvec(vec, nrow, ncol):
    return vec.reshape(nrow, ncol, order='F')

def vec(matrix):
    return matrix.flatten(order='F').reshape(-1, 1)

def LSKRF(matrix, I, J, R):
    # X: A matrix created by the relation X = A ⋄ B ∈ C^{IJ×R}, where A ∈ C^{I×R} and B ∈ C^{J×R}.
    # (I,J,R): Dimensions of A and B as mentioned above.

    # Creating a matrix to alocate the values of stimated X.
    
    A_hat = np.zeros((I,R), dtype=complex)
    B_hat = np.zeros((J,R), dtype=complex)

    for i in range(0, R):
        # Each column of X will be rearranged into a matrix Xp ∈ ℂ^{J×I}.
        matrix_p = (matrix[:, i]).reshape(J, I, order='F')

        # In this line the SVD of the new matrix will be calculated.
        [U_p, S_p, V_p] = np.linalg.svd(matrix_p)

        # Now the stimations of the a_hat and b_hat i-th columns will be made.
        a_hat = np.sqrt(S_p[0]) * ((V_p[0, :]))
        b_hat = np.sqrt(S_p[0]) * U_p[:, 0]

        # Finally, the kronecker between the stimations of the i-th columns of a_hat and b_hat
        # will be calculated and alocated to the respective column of the stimation of X.
        A_hat[:, i] = a_hat
        B_hat[:, i] = b_hat

    return A_hat, B_hat

def LSKronF(X, I, P, J, Q):
    # Reshape para resolução via SVD
    X_reshaped = X.reshape(I, J, P, Q).transpose(0, 2, 1, 3).reshape(I*P, J*Q)
    U, S, Vh = np.linalg.svd(X_reshaped, full_matrices=False)
    rank1 = np.sqrt(S[0]) * U[:, 0], np.sqrt(S[0]) * Vh[0, :]
    
    Ahat = rank1[0].reshape(I, P)
    Bhat = rank1[1].reshape(J, Q)
    return Ahat, Bhat

# para o problema 1
    
A = np.random.randn(4, 2) + 1j*np.random.randn(4, 2)
B = np.random.randn(6, 3) + 1j*np.random.randn(6, 3)
X = np.kron(A, B)
print(X.shape)
Ahat, Bhat = LSKronF(X, 4, 2, 6, 3)
Xhat = np.kron(Ahat, Bhat)
print(Xhat.shape)

print("Checking the NMSE (dB) between X and Xhat:")
nmsex = (np.linalg.norm(X - Xhat, 'fro')**2) / (np.linalg.norm(X, 'fro')**2)
nmsex = 20 * np.log10(nmsex)
print(nmsex)

print("Checking the NMSE (dB) between A and Ahat:")
nmsea = (np.linalg.norm(A - Ahat, 'fro')**2) / (np.linalg.norm(A, 'fro')**2)
nmsea = 20 * np.log10(nmsea)
print(nmsea)

print("Checking the NMSE (dB) between B and Bhat:")
nmseb = (np.linalg.norm(B - Bhat, 'fro')**2) / (np.linalg.norm(B, 'fro')**2)
nmseb = 20 * np.log10(nmseb)
print(nmseb)

# para o problema 2
def simula_nmse(I, J, R, SNR_vals):
    nmse = np.zeros(len(SNR_vals))
    for idx, snr in enumerate(SNR_vals):
        for _ in range(1000):
            var_noise = 1 / (10**(snr / 10))
            noise = np.sqrt(var_noise / 2) * (np.random.randn(I*J, R) + 1j*np.random.randn(I*J, R))

            A = np.random.randn(I, R) + 1j*np.random.randn(I, R)
            B = np.random.randn(J, R) + 1j*np.random.randn(J, R)
            X = khatri_rao(A, B)
            X_noisy = X + noise

            Ahat, Bhat = LSKRF(X_noisy, I, J, R)
            Xhat = khatri_rao(Ahat, Bhat)

            aux = (np.linalg.norm(X - Xhat, 'fro')**2) / (np.linalg.norm(X, 'fro')**2)
            nmse[idx] += 20 * np.log10(aux)
    return nmse / 1000

SNR = np.array([0, 5, 10, 15, 20, 25, 30])

nmse1 = simula_nmse(I=10, J=10, R=4, SNR_vals=SNR)
nmse2 = simula_nmse(I=30, J=10, R=4, SNR_vals=SNR)

# plot
plt.figure()
plt.plot(SNR, nmse1, label=f"I = 2, J = 4, P = 3, Q = 5")
plt.plot(SNR, nmse2, label=f"I = 2, J = 4, P = 3, Q = 5")
plt.title("LSKronF performance under imperfect scenario") 
plt.xlabel("SNR (dB)")
plt.ylabel("NMSE (dB)")
plt.legend()
plt.grid(True)
#plt.savefig("hw3.pdf")
plt.show()  