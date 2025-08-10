import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from numpy.linalg import svd
from skimage import img_as_float, img_as_ubyte

img = img_as_float(imread('C:\\Users\\Ruan Pereira\\Documents\\GitHub\\tensores-study\\homeworklin03\\image\\image\\svd.jpg', pilmode='L'))  # 'L' for grayscale
X = np.array(img, dtype=float)
M, N = X.shape

U, s, Vt = svd(X, full_matrices=False) 
energy = s**2
cum_energy = np.cumsum(energy) / np.sum(energy)

# Plot valores singulares e energia cumulativa
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.semilogy(s, '.-'); plt.title('Singular values'); plt.xlabel('index'); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(cum_energy, '.-'); plt.title('Cumulative energy'); plt.xlabel('J'); plt.axhline(0.95, color='k', linestyle='--')
plt.grid(True); plt.show()

# usando um threshold de 99% de energia
threshold = 0.99
J = np.searchsorted(cum_energy, threshold) + 1
print("Chosen J for {:.1f}% energy:".format(threshold*100), J)

def reconstruct_from_svd(U, s, Vt, J):
    return (U[:, :J] * s[:J]) @ Vt[:J, :]

XJ = reconstruct_from_svd(U, s, Vt, J)

# imagem reconstruida
imwrite('recon_J.pdf', img_as_ubyte(np.clip(XJ,0,1)))

def add_awgn(X, snr_db):
    M, N = X.shape
    signal_energy = np.linalg.norm(X, 'fro')**2 
    snr_linear = 10**(snr_db/10.0)
    sigma = np.sqrt(signal_energy / (M*N*snr_linear))
    Z = np.random.normal(scale=sigma, size=X.shape)
    return X + Z, sigma

snr_db = 20
Y, sigma = add_awgn(X, snr_db)
print("Noise sigma per entry:", sigma)

# SVD na imagem ruidosa
U_y, s_y, Vt_y = svd(Y, full_matrices=False)
cum_energy_y = np.cumsum(s_y**2) / np.sum(s_y**2)


def mse(A,B): return np.mean((A-B)**2)

Js = list(range(1, min(M,N)+1, max(1, min(M,N)//100)))
mses = []
for j in Js:
    Yj = reconstruct_from_svd(U_y, s_y, Vt_y, j)
    mses.append(mse(X, Yj))

# Plot MSE vs J
plt.figure()
plt.plot(Js, mses, '.-')
plt.xlabel('J'); plt.ylabel('MSE vs clean X'); plt.title(f'MSE (noisy SNR={snr_db} dB)')
plt.grid(True); plt.show()

# reconstrucao com J escolhido e ruido
J_example = J
YJ_example = reconstruct_from_svd(U_y, s_y, Vt_y, J_example)
imwrite('noisy_recon_J.pdf', img_as_ubyte(np.clip(YJ_example,0,1)))


def psnr(X, Xhat, maxval=1.0):
    msev = np.mean((X - Xhat)**2)
    if msev == 0: return float('inf')
    return 10*np.log10(maxval**2 / msev)

print("PSNR clean rank-J:", psnr(X, XJ))
print("PSNR noisy rank-J:", psnr(X, YJ_example))