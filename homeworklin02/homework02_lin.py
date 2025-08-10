import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Para o problema 1
# Nome do arquivo .mat
filename = 'C:\\Users\\Ruan Pereira\\Documents\\GitHub\\tensores-study\\homeworklin02\\Homework_02_data.mat'

# Carrega os dados do arquivo .mat
mat_data = scipy.io.loadmat(filename)

# Set global font size
#plt.rcParams.update({'font.size': 16})

x = np.arange(-10, 11).reshape(-1, 1)

# loading data from .mat
y_linear = mat_data['observation_linear']
y_quadratic = mat_data['observation_quadratic']
y_cubic = mat_data['observation_cubic']

# Create projection matrices
projection_mtx_1 = np.hstack([np.ones((len(x), 1)), x])
projection_mtx_2 = np.hstack([np.ones((len(x), 1)), x, x**2])
projection_mtx_3 = np.hstack([np.ones((len(x), 1)), x, x**2, x**3])

# Compute solutions
solution_1 = np.linalg.pinv(projection_mtx_1) @ y_linear
solution_2 = np.linalg.pinv(projection_mtx_2) @ y_quadratic
solution_3 = np.linalg.pinv(projection_mtx_3) @ y_cubic

# Create estimated curves
y_linear_est = solution_1[0] + solution_1[1]*x
y_quadratic_est = solution_2[0] + solution_2[1]*x + solution_2[2]*(x**2)
y_cubic_est = solution_3[0] + solution_3[1]*x + solution_3[2]*(x**2) + solution_3[3]*(x**3)

# Plot linear results
plt.figure(figsize=(10, 6))
plt.plot(x, y_linear_est, '-o', color='#0072BD', linewidth=3, markersize=8, label='Estimated vector')
plt.plot(x, y_linear, 'd', color='#D95319', linewidth=3, markersize=8, label='Noisy vector')
plt.ylabel('SE($\\hat{y}$)', fontsize=20)
plt.legend(loc='best')
plt.grid(True)
plt.title('Linear Estimation')
plt.show()

# Plot quadratic results
plt.figure(figsize=(10, 6))
plt.plot(x, y_quadratic_est, '-o', color='#0072BD', linewidth=3, markersize=8, label='Estimated vector')
plt.plot(x, y_quadratic, 'd', color='#D95319', linewidth=3, markersize=8, label='Noisy vector')
plt.ylabel('SE($\\hat{y}$)', fontsize=20)
plt.legend(loc='best')
plt.grid(True)
plt.title('Quadratic Estimation')
plt.show()

# Plot cubic results
plt.figure(figsize=(10, 6))
plt.plot(x, y_cubic_est, '-o', color='#0072BD', linewidth=3, markersize=8, label='Estimated vector')
plt.plot(x, y_cubic, 'd', color='#D95319', linewidth=3, markersize=8, label='Noisy vector')
plt.ylabel('SE($\\hat{y}$)', fontsize=20)
plt.legend(loc='best')
plt.grid(True)
plt.title('Cubic Estimation')
plt.show()

# Para o problema 2
SNR_db = np.arange(0, 31, 5)
iter_max = 100
error = np.zeros((len(SNR_db), iter_max))

for snr_idx, snr_val in enumerate(SNR_db):
    for iter in range(iter_max):
        x = np.random.randn(5, 1) + 1j*np.random.randn(5, 1)
        A = np.random.randn(10, 5) + 1j*np.random.randn(10, 5)
        y = A @ x
        
        snr_linear = 10**(snr_val/10)
        var_noise = (np.linalg.norm(y)**2) / snr_linear
        noise = np.sqrt(var_noise/2) * (np.random.randn(10, 1) + 1j*np.random.randn(10, 1))
        
        y_noisy = y + noise
        x_est = np.linalg.pinv(A) @ y_noisy
        
        nmse = np.linalg.norm(y - A @ x_est)**2 / np.linalg.norm(y)**2
        error[snr_idx, iter] = nmse

mean_error = 10 * np.log10(np.mean(error, axis=1))

plt.figure(figsize=(10, 6))
plt.plot(SNR_db, mean_error, '-o', label='Least Squares')
plt.ylabel('NMSE($\\hat{y}$) in dB')
plt.xlabel('SNR (dB)')
plt.legend(loc='best')
plt.grid(True)
plt.title('Least Squares Performance')
plt.show()

# Para o problema 3
M = 100
N = np.arange(100, 1001, 100)
cost = np.zeros(len(N))

for i, n_val in enumerate(N):
    cost[i] = max(M, n_val) * min(M, n_val)**2

plt.figure(figsize=(10, 6))
plt.semilogy(N, cost, '-o', color='#0072BD', linewidth=3, markersize=8)
plt.ylabel('Computational Cost (FLOPS)')
plt.xlabel('Number of Columns (N)')
plt.grid(True)
plt.title('Pseudoinverse Computational Complexity')
plt.xlim(100, 1000)
plt.show()