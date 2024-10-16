# aggregation.py
import numpy as np
def aggregate_models(global_model, local_models, selected_clients, data, num_modalities):
    global_weights = np.random.rand(num_modalities)

    local_model_predictions = []
    for client in selected_clients:
        client_predictions = []
        for modality in range(num_modalities):
            prediction = local_models[client].predict(data[client], modality)
            if len(prediction) != len(data[client]):
                raise ValueError(f"Inconsistent number of samples for client {client} and modality {modality}")
            client_predictions.append(prediction)
        local_model_predictions.append(client_predictions)

    global_model.aggregate_models(local_model_predictions, global_weights)
    return global_model.get_global_models()