# main.py
import numpy as np
from mmfedmc.local_model import LocalEnsembleModel
from mmfedmc.global_model import GlobalModel
from mmfedmc.communication import Communication
from mmfedmc.fusion import DecisionFusion
from config import T, num_clients, num_modalities, E, learning_rate, alpha_s, alpha_c, gamma, delta
from data_generation import generate_data
from training import train_local_models
from selection import select_modalities_and_clients
from aggregation import aggregate_models
from generalization_error import calculate_ge, calculate_geb, logistic_loss

# Generate data
data, labels, local_losses = generate_data(num_clients)

# Initialize models
local_models = [LocalEnsembleModel(num_modalities) for _ in range(num_clients)]
global_model = GlobalModel(num_modalities)
communication = Communication(num_modalities)
fusion = DecisionFusion()

# Global iterations
for t in range(T):
    print(f"通信轮次 {t+1}")

    # Local training
    train_local_models(local_models, data, labels, num_modalities, E, learning_rate)

    # Modality and client selection
    selected_modalities, selected_clients = select_modalities_and_clients(communication, local_models, data, labels, gamma, local_losses, delta)

    # Model aggregation
    global_models = aggregate_models(global_model, local_models, selected_clients, data, num_modalities)

    # Local deployment
    for client_index, local_model in enumerate(local_models):
        local_model.update_ensemble_model_stage_2(global_models, data[client_index], labels[client_index])
    for client_index, local_model in enumerate(local_models):
        model_sizes = local_model.get_model_sizes()
    print(f"Client {client_index} model sizes: {model_sizes}")

# Example GE and GEB calculations
example_f = lambda x: np.dot(x, np.random.rand(x.shape[1]))  # a dummy model function
example_D = list(zip(data[0], labels[0]))  # a dummy dataset
empirical_errors = [calculate_ge(example_f, example_D, logistic_loss) for _ in range(num_modalities)]
fusion_weights = np.random.rand(num_modalities, num_modalities)
losses = np.random.rand(num_modalities, num_modalities)

geb = calculate_geb(range(num_modalities), 0.1, delta, len(data[0]), empirical_errors, fusion_weights, losses)
print(f"Generalization Error Bound: {geb}")

# Update and save final models
final_global_models = global_model.get_global_models()
final_local_models = [local_model.save_model() for local_model in local_models]

print("算法运行完成，模型已更新。")