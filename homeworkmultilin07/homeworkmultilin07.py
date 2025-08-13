import numpy as np
import time
import matplotlib.pyplot as plt

import scipy.io 

mat_data = scipy.io.loadmat('homeworkmultilin07/Practice_7_krf_matrix_3D.mat')

ten_X = mat_data['X']
A = mat_data['A']
B = mat_data['B']
C = mat_data['C']

def MLSKRF(X,I,R):
    
    A = np.zeros((I[0], R))
    B = np.zeros((I[1], R)) 
    C = np.zeros((I[2], R))   

    for i in range(R):

        X_r = X[:,i]
        X_r = fold(X_r, I, 1)

        modes = len(X_r.shape)
        
        for i in range(modes):
            tensor_unfolded = unfold(X_r, i+1)
            [u, _, _] = np.linalg.svd(tensor_unfolded, full_matrices=False)
            s_tensor = ten_mode_prod(X_r, u.T, i+1) 

            if i == 0:
                A[:, i] = np.sqrt(np.max(s_tensor))**(2/len(I)) * u[:,0]
            elif i == 1:    
                B[:, i] = np.sqrt(np.max(s_tensor))**(2/len(I)) * u[:,0]
            else:    
                C[:, i] = np.sqrt(np.max(s_tensor))**(2/len(I)) * u[:,0]
        
    return A,B,C

def unfold(tensor, n):
    n = n - 1
    tensor = np.moveaxis(tensor, n, 0)
    tensor_unfolding = tensor.reshape(tensor.shape[0], -1)

    return tensor_unfolding

def fold(tensor_unfolding, tensor_shape, n):
    n = n - 1
    # Transforming the shape of tensor tuple into a list for easy manipulation.
    shape = list(tensor_shape)
    # Extracting the external dimension that is presented in the unfolding tensor as the number of rows.
    n_dimension = shape.pop(n)
    # Inserting the previously dimension at the begining of the shape vector so this way we have a dinamic reshape
    # that will change in accord with the unfolding mode.
    shape.insert(0, n_dimension)

    # Reorganizing the unfolded tensor as a tensor.
    tensor = tensor_unfolding.reshape(shape)

    # Moving back the axis that was changed at the unfolding function.
    tensor = np.moveaxis(tensor, 0, n)

    return tensor


def ten_mode_prod(tensor, matrix, n):
    
    shape = list(tensor.shape)
    shape[n-1] = matrix.shape[0]

    tensor = matrix@ unfold(tensor,n)
    tensor = fold(tensor, shape, n)

    return tensor

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

aux = khatri_rao(A,B)
X = khatri_rao(aux, C)

I = [A.shape[0], B.shape[0], C.shape[0]]
print(I)

R = X.shape[1]

A_hat,B_hat,C_hat = MLSKRF(X, I, R)


print("Shape of A:", A.shape)
print("Shape of B:", B.shape)
print("Shape of C:", C.shape)


# para o problema 2

import matplotlib.pyplot as plt

def generate_noisy_data(I, R, snr_db):
    # Generate random factor matrices
    A = np.random.randn(I[0], R) + 1j*np.random.randn(I[0], R)
    B = np.random.randn(I[1], R) + 1j*np.random.randn(I[1], R)
    C = np.random.randn(I[2], R) + 1j*np.random.randn(I[2], R)

    # Create noiseless data
    X0 = khatri_rao(A, khatri_rao(B, C))
    
    # Generate noise
    V = np.random.randn(*X0.shape) + 1j*np.random.randn(*X0.shape)
    
    # Calculate alpha based on SNR
    snr_linear = 10**(snr_db / 10)
    alpha = np.sqrt(np.linalg.norm(X0, 'fro')**2 / (snr_linear * np.linalg.norm(V, 'fro')**2))
    
    # Create noisy data
    X = X0 + alpha * V
    
    return X0, X, A, B, C

def monte_carlo_experiment(I, R, snr_range, num_experiments=1000):
    nmse_results = []
    
    for snr in snr_range:
        total_nmse = 0
        
        for _ in range(num_experiments):
            # Generate noisy data
            X0, X, A_true, B_true, C_true = generate_noisy_data(I, R, snr)
            
            # Estimate factor matrices
            A_est, B_est, C_est = MLSKRF(X, I, R)
            
            # Reconstruct X0
            X0_est = khatri_rao(A_est, khatri_rao(B_est, C_est))
            
            # Calculate NMSE for this experiment
            nmse = np.linalg.norm(X0_est - X0, 'fro')**2 / np.linalg.norm(X0, 'fro')**2
            total_nmse += nmse
        
        # Average NMSE over all experiments
        avg_nmse = total_nmse / num_experiments
        nmse_results.append(avg_nmse)
    
    return nmse_results

# Parameters
R = 4
snr_range = [0, 5, 10, 15, 20, 25, 30]  # in dB

# Run Monte Carlo experiments
nmse_values = monte_carlo_experiment(I, R, snr_range)

# Plot results
plt.figure(figsize=(8, 5))
plt.semilogy(snr_range, nmse_values, 'bo-')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')
plt.title('NMSE vs SNR for MLS-KRF Algorithm')
plt.grid(True, which="both", ls="-")
plt.show()