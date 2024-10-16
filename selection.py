# selection.py
def select_modalities_and_clients(communication, local_models, data, labels, gamma, local_losses, delta):
    shapley_values = [communication.compute_shapley_values([local_model.predict(data[client_index], i) for i in range(num_modalities)], labels[client_index]) for client_index in range(num_clients)]
    selected_modalities = [communication.select_modalities(shapley_values[client_index], np.random.rand(), np.random.rand(), gamma) for client_index in range(num_clients)]
    selected_clients = communication.select_clients(local_losses, delta)
    return selected_modalities, selected_clients