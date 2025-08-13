import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import kron, svd
from scipy.io import loadmat
import os

def mls_kronf(X, dims):
    """
    Multidimensional Least-Squares Kronecker Factorization (MLS-KronF) for 3D case
    
    Parameters:
    X : ndarray
        The matrix to be factorized
    dims : list of tuples
        Dimensions of the factor matrices [(I1, J1), (I2, J2), (I3, J3)]
    
    Returns:
    tuple of ndarrays
        Estimated factor matrices (A, B, C)
    """
    (I1, J1), (I2, J2), (I3, J3) = dims
    
    # Step 1: Reshape X for first factorization
    X_reshaped = X.reshape(I1, I2*I3, J1, J2*J3)
    X_reshaped = X_reshaped.transpose(0, 2, 1, 3)
    X_reshaped = X_reshaped.reshape(I1*J1, I2*I3*J2*J3)
    
    # Step 2: SVD for first factor (A)
    U, S, Vh = svd(X_reshaped, full_matrices=False)
    A = U[:, 0].reshape(I1, J1)
    
    # Step 3: Reshape for second factorization
    X_proj = (U[:, 0].conj().T @ X_reshaped).reshape(I2*J2, I3*J3)
    
    # Step 4: SVD for second factor (B)
    U, S, Vh = svd(X_proj, full_matrices=False)
    B = U[:, 0].reshape(I2, J2)
    
    # Step 5: Get third factor (C)
    C = Vh[0, :].reshape(I3, J3)
    
    # Normalize factors
    norm_A = np.linalg.norm(A, 'fro')
    norm_B = np.linalg.norm(B, 'fro')
    norm_C = np.linalg.norm(C, 'fro')
    
    A = A / norm_A * (norm_A * norm_B * norm_C)**(1/3)
    B = B / norm_B * (norm_A * norm_B * norm_C)**(1/3)
    C = C / norm_C * (norm_A * norm_B * norm_C)**(1/3)
    
    return A, B, C

def monte_carlo_experiment(I1, J1, I2, J2, I3, J3, snr_db_values, num_experiments=1000):
    """
    Perform Monte Carlo experiments for different SNR values
    
    Returns:
    dict
        NMSE values for each SNR
    """
    nmse_results = {snr: [] for snr in snr_db_values}
    
    for snr_db in snr_db_values:
        for _ in range(num_experiments):
            # Generate random factor matrices
            A = np.random.randn(I1, J1) + 1j*np.random.randn(I1, J1)
            B = np.random.randn(I2, J2) + 1j*np.random.randn(I2, J2)
            C = np.random.randn(I3, J3) + 1j*np.random.randn(I3, J3)
            
            # Create noiseless X0
            X0 = kron(kron(A, B), C)
            
            # Create noise matrix
            V = np.random.randn(*X0.shape) + 1j*np.random.randn(*X0.shape)
            
            # Calculate alpha based on SNR
            snr_linear = 10**(snr_db / 10)
            alpha = np.sqrt(np.linalg.norm(X0, 'fro')**2 / (snr_linear * np.linalg.norm(V, 'fro')**2))
            
            # Create noisy X
            X = X0 + alpha * V
            
            # Apply MLS-KronF
            dims = [(I1, J1), (I2, J2), (I3, J3)]
            A_hat, B_hat, C_hat = mls_kronf(X, dims)
            
            # Reconstruct X0_hat
            X0_hat = kron(kron(A_hat, B_hat), C_hat)
            
            # Calculate NMSE
            nmse = np.linalg.norm(X0_hat - X0, 'fro')**2 / np.linalg.norm(X0, 'fro')**2
            nmse_results[snr_db].append(nmse)
    
    # Calculate average NMSE for each SNR
    for snr in nmse_results:
        nmse_results[snr] = np.mean(nmse_results[snr])
    
    return nmse_results

def validate_with_matfile():
    """
    Validate the implementation with the provided .mat file
    """
    try:
        # Load the .mat file (adjust path as needed)
        mat_data = loadmat('kronf_matrix_3D.mat')
        X = mat_data['X']
        
        # Assuming we know the dimensions (adjust as needed)
        I1, J1 = 2, 2
        I2, J2 = 3, 3
        I3, J3 = 4, 4
        
        dims = [(I1, J1), (I2, J2), (I3, J3)]
        A_hat, B_hat, C_hat = mls_kronf(X, dims)
        
        # Reconstruct X
        X_hat = kron(kron(A_hat, B_hat), C_hat)
        
        # Calculate reconstruction error
        error = np.linalg.norm(X - X_hat, 'fro') / np.linalg.norm(X, 'fro')
        print(f"Reconstruction relative error: {error:.4f}")
        
        return A_hat, B_hat, C_hat, error
    except FileNotFoundError:
        print("kronf_matrix_3D.mat file not found. Please ensure it's in your working directory.")
        return None, None, None, None

# Main experiment
if __name__ == "__main__":
    # Parameters
    I1, J1 = 2, 2
    I2, J2 = 3, 3
    I3, J3 = 4, 4
    snr_db_values = [0, 5, 10, 15, 20, 25, 30]
    
    # Run Monte Carlo experiments
    nmse_results = monte_carlo_experiment(I1, J1, I2, J2, I3, J3, snr_db_values)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(list(nmse_results.keys()), list(nmse_results.values()), 'bo-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (log scale)')
    plt.title('MLSKronF NMSE vs. SNR: I$_1$=J$_1$ = 2, I$_2$ = J$_2$= 3, I$_3$=J$_3$ =4')
    plt.grid(True, which="both", ls="-")
    plt.show()
    
    # Validate with .mat file if available
    A_hat, B_hat, C_hat, error = validate_with_matfile()