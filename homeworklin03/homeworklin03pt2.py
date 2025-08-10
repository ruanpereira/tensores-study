import numpy as np
import scipy.io as sio
import mat73
import matplotlib.pyplot as plt

mat_data = mat73.loadmat('C:\\Users\\Ruan Pereira\\Documents\\GitHub\\tensores-study\\homeworklin03\\tensor\\tensor\\ten_H.mat')
H_clean = mat_data['ten_H']

# parametros dados na questao
R_candidates = np.arange(1, 10)
nMC = 1000
SNR_dB = 20
maxIter = 100

mean_final_err = np.zeros(len(R_candidates))
std_final_err  = np.zeros(len(R_candidates))
mean_iter_err  = []

def add_noise_to_tensor(H_clean, SNR_dB):
    signal_energy = np.linalg.norm(H_clean)**2
    snr_linear = 10**(SNR_dB/10)
    sigma = np.sqrt(signal_energy / (H_clean.size * snr_linear))
    noise = sigma * np.random.randn(*H_clean.shape)
    return H_clean + noise

for idx, Rhat in enumerate(R_candidates):
    final_errs = np.zeros(nMC)
    iter_traces = np.zeros((nMC, maxIter))

    for mc in range(nMC):
        H_noisy = add_noise_to_tensor(H_clean, SNR_dB)
        y, final_err = ALS_estimation(H_noisy, Rhat, SNR_dB)
        final_errs[mc] = final_err

        L = min(len(y), maxIter)
        iter_traces[mc, :L] = y[:L]
        if L < maxIter:
            iter_traces[mc, L:] = y[-1]

    mean_final_err[idx] = np.mean(final_errs)
    std_final_err[idx] = np.std(final_errs)
    mean_iter_err.append(np.mean(iter_traces, axis=0))

    print(f"Rhat={Rhat} -> mean final err={mean_final_err[idx]:.4e}, "
          f"std={std_final_err[idx]:.4e}")

# plot do erro por interacao de ALS para os Ranks
plt.figure()
to_plot = [1, R_candidates[len(R_candidates)//2], R_candidates[-1]]
for R in to_plot:
    idx = np.where(R_candidates == R)[0][0]
    plt.plot(mean_iter_err[idx], lw=1.2, label=fr'$\hat{{R}}$={R}')

plt.legend()
plt.xlabel('ALS iteration')
plt.ylabel('Mean error')
plt.grid(True)
plt.title('Mean ALS iteration error for selected $\hat{R}$')

plt.show()