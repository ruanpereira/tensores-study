import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import kron, svd
from scipy.io import loadmat

mat_data = loadmat('homeworkmultilin09\Practice_9_cpd_tensor.mat')
A = mat_data['A']
print(f'A original shape: {A.shape}')
B = mat_data['B']
print(f'B original shape: {B.shape}')
C = mat_data['C']
print(f'C original shape: {C.shape}')
tenX = mat_data['tenX']

def unfold(tensor, n):
    n = n - 1
    tensor = np.moveaxis(tensor, n, 0)
    tensor_unfolding = tensor.reshape(tensor.shape[0], -1)

    return tensor_unfolding

def fold(tensor_unfolding, tensor_shape, n):
    n = n-1
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

def ALS(X, R):
    ia, ib, ic = X.shape

    # Inicializações aleatórias complexas
    Ahat = np.random.randn(ia, R) + 1j * np.random.randn(ia, R)
    Bhat = np.random.randn(ib, R) + 1j * np.random.randn(ib, R)
    Chat = np.random.randn(ic, R) + 1j * np.random.randn(ic, R)

    # Unfoldings
    mode_1 = unfold(X, 1)
    mode_2 = unfold(X, 2)
    mode_3 = unfold(X, 3)

    aux = 1000
    error = np.zeros(aux, dtype=float)

    # Erro inicial
    error[0] = (np.linalg.norm(mode_1 - Ahat @ khatri_rao(Chat, Bhat).T, 'fro') ** 2) / \
               (np.linalg.norm(mode_1, 'fro') ** 2)

    # Iterações ALS
    for i in range(1, aux):
        Bhat = mode_2 @ np.linalg.pinv(khatri_rao(Chat, Ahat).T)
        Chat = mode_3 @ np.linalg.pinv(khatri_rao(Bhat, Ahat).T)
        Ahat = mode_1 @ np.linalg.pinv(khatri_rao(Chat, Bhat).T)

        error[i] = (np.linalg.norm(mode_1 - Ahat @ khatri_rao(Chat, Bhat).T, 'fro') ** 2) / \
                   (np.linalg.norm(mode_1, 'fro') ** 2)

        if abs(error[i] - error[i - 1]) < np.finfo(float).eps:
            error = error[:i + 1]
            break

    return Ahat, Bhat, Chat, error

def normalized_mean_square_error(tensor, tensor_hat):
    nmse = (np.linalg.norm(tensor - tensor_hat)) ** 2 / (np.linalg.norm(tensor)) ** 2

    return nmse

# para o problema 1
A_est, B_est, C_est, error = ALS(tenX, 3)

print(f'A_est: {A_est}')
print(f'B_est: {B_est}')
print(f'C_est: {C_est}')
print(f'Error: {error}')


print(f'NMSE A: {normalized_mean_square_error(A, A_est)}')
print(f'NMSE B: {normalized_mean_square_error(B, B_est)}')
print(f'NMSE C: {normalized_mean_square_error(C, C_est)}')

# para o problema 2
def generate_noisy_data(I, J, K, R, snr_db):
    # Generate random factor matrices
    A = np.random.randn(I, R) + 1j*np.random.randn(I, R)
    B = np.random.randn(J, R) + 1j*np.random.randn(J, R)
    C = np.random.randn(K, R) + 1j*np.random.randn(K, R)
    print(A.shape)
    print(B.shape)
    print(C.shape)

    # Create noiseless data
    X0 = fold(A @ (khatri_rao(C,B)).T, [I,J,K], 1)
    
    # Generate noise
    V = np.random.randn(*X0.shape) + 1j*np.random.randn(*X0.shape)
    
    # Calculate alpha based on SNR
    snr_linear = 10**(snr_db / 10)
    alpha = np.sqrt(np.linalg.norm(X0, 'fro')**2 / (snr_linear * np.linalg.norm(V, 'fro')**2))
    
    # Create noisy data
    X = X0 + alpha * V
    
    return X0, X, A, B, C

def monte_carlo_experiment(I, J, K, R, snr_range, num_experiments=1000):
    nmse_results = []
    
    for snr in snr_range:
        total_nmse = 0
        
        for _ in range(num_experiments):
            # Generate noisy data
            X0, X, A_true, B_true, C_true = generate_noisy_data(I, J, K, R, snr)
            print(f'X0.shape: {X0.shape}')

            # Estimate factor matrices
            A_est, B_est, C_est, error = ALS(X, R)

            # Reconstruct X0
            X0_unfolded = A_est @ khatri_rao(C_est, B_est).T
            tenX_hat = fold(X0_unfolded, [I, J, K], 1)
            # Calculate NMSE for this experiment
            nmse = np.linalg.norm(tenX_hat - X0, 'fro')**2 / np.linalg.norm(X0, 'fro')**2
            total_nmse += nmse
        
        # Average NMSE over all experiments
        avg_nmse = total_nmse / num_experiments
        nmse_results.append(avg_nmse)
    
    return nmse_results

# Parameters
I, J, K, R = 10, 4, 2, 3
snr_range = [0, 5, 10, 15, 20, 25, 30]  # in dB

# Run Monte Carlo experiments
nmse_values = monte_carlo_experiment(I, J, K, R, snr_range)

# Plot results
plt.figure(figsize=(8, 5))
plt.semilogy(snr_range, nmse_values, 'bo-')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (log scale)')
plt.title('NMSE vs SNR for MLS-KRF Algorithm')
plt.grid(True, which="both", ls="-")
plt.show()
