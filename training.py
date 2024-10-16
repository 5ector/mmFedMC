# training.py
def train_local_models(local_models, data, labels, num_modalities, E, learning_rate):
    for client_index, local_model in enumerate(local_models):
        for modality_index in range(num_modalities):
            local_model.train(data[client_index], labels[client_index], modality_index, epochs=E, learning_rate=learning_rate)
        predictions = [local_model.predict(data[client_index], i) for i in range(num_modalities)]
        local_model.update_ensemble_model(predictions, labels[client_index], epochs=E, learning_rate=learning_rate)