# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 20:18:44 2025

@author: majela.penton
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import data
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import img_as_float
from skimage.transform import resize
from time import time
import seaborn as sns
import pandas as pd

from utilities_tv import padPSF, kronDecomp
from algorithms import PepsilonSM

# Load ground truth image
X = img_as_float(data.camera())
# Resize image 
x_opt = resize(X, (128, 128), anti_aliasing=True)

# Define blur kernel
P = 1/9 * np.ones((3, 3))
Pbig = padPSF(P, x_opt.shape)
Ar, Ac = kronDecomp(Pbig, [1, 1], 'reflexive')

# Generate blurred + noisy observation
np.random.seed(314)
b = convolve(x_opt, P, mode='reflect') + 1e-4 * np.random.randn(*x_opt.shape)

# Parameters for PeSM
sr = np.linalg.svd(Ar, compute_uv=False)
sc = np.linalg.svd(Ac, compute_uv=False)
Sbig = np.outer(sc, sr)
L = 2 * np.max(Sbig**2)
alpha = 1 / L
tau = 1e-4
maxIt = 1000
tol = 1e-4
x0 = b


# Ranges to compare
sigma_list = [0.1, 0.3, 0.5, 0.7, 0.9]

# Collecting the results
results = []

for sigma in sigma_list:
    # Run PeSM
    start = time()
    x_psm, f_psm, _, _, num_inner_iterations = PepsilonSM(sigma, x0, x_opt, Ac, Ar, b, alpha, tau, maxIt, tol)
    time_psm = time() - start
    psnr_psm = peak_signal_noise_ratio(x_opt, x_psm)
    ssim_psm = structural_similarity(x_opt, x_psm, data_range=1.0)

    results.append({
        'sigma': sigma,
        'time': time_psm,
        'outer_iterations': len(f_psm),
        'inner_iterations': num_inner_iterations,
        'PSNR': psnr_psm,
        'SSIM': ssim_psm
    })



# --- Saving the results ---
df = pd.DataFrame(results)

df.to_csv('results/performance.csv', index=False)

# --- Plotting ---
# Metrics to compare
metrics = ['time', 'outer_iterations', 'inner_iterations', 'PSNR', 'SSIM']

# Compare PeSM for varying sigma
for metric in metrics:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='sigma', y=metric, palette='viridis')
    plt.title(f"PeSM - {metric.upper()} vs Sigma")
    plt.xlabel('Sigma')
    plt.ylabel(metric)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'plots/barplot_{metric}_vs_sigma.png')
    plt.close()
        
        

