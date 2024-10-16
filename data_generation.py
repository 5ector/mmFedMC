# data_generation.py
import numpy as np

def generate_data(num_clients):
    data = [np.random.rand(100, 10) for _ in range(num_clients)]
    labels = [np.random.randint(0, 2, 100) for _ in range(num_clients)]
    local_losses = [np.random.rand() for _ in range(num_clients)]
    return data, labels, local_losses