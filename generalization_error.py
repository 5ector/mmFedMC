# generalization_error.py
import numpy as np

def calculate_ge(f, D, loss_fn):
    return np.mean([loss_fn(f(x), y) for x, y in D])

def calculate_geb(modalities, R_N_H, delta, N, empirical_errors, fusion_weights, losses):
    M = len(modalities)
    term1 = M * (R_N_H + np.sqrt(np.log(1 / delta) / (2 * N)))
    term2 = np.sum(empirical_errors)
    
    cov_mono = np.array([np.cov(fusion_weights[m], losses[m])[0][1] for m in range(M)])
    cov_holo = np.array([[np.cov(fusion_weights[m], losses[j])[0][1] for j in range(M) if j != m] for m in range(M)])
    
    term3 = np.sum(cov_mono) / M
    term4 = np.sum(cov_holo) * (M - 1) / M
    
    return term1 + term2 + term3 - term4

# Define a sample logistic loss function
def logistic_loss(pred, true):
    return np.log(1 + np.exp(-true * pred))