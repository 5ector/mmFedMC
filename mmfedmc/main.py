import numpy as np
from local_model.local_model import LocalEnsembleModel
from global_model.global_model import GlobalModel
from communication.communication import Communication
from fusion.fusion import DecisionFusion

# 初始化参数
T = 10  # 通信轮次
num_clients = 5
num_modalities = 3
E = 5  # 本地训练轮次
learning_rate = 0.01
alpha_s = 0.5
alpha_c = 0.5
gamma = 2
delta = 2

# 生成模拟数据
data = [np.random.rand(100, 10) for _ in range(num_clients)]
labels = [np.random.randint(0, 2, 100) for _ in range(num_clients)]
local_losses = [np.random.rand() for _ in range(num_clients)]

# 初始化模型
local_models = [LocalEnsembleModel(num_modalities) for _ in range(num_clients)]
global_model = GlobalModel(num_modalities)
communication = Communication(num_modalities)
fusion = DecisionFusion()

# 全局迭代
for t in range(T):
    print(f"通信轮次 {t+1}")

    # 本地学习
    for client_index, local_model in enumerate(local_models):
        for modality_index in range(num_modalities):
            local_model.train(data[client_index], labels[client_index], modality_index, epochs=E, learning_rate=learning_rate)
        predictions = [local_model.predict(data[client_index], i) for i in range(num_modalities)]
        local_model.update_ensemble_model(predictions, labels[client_index], epochs=E, learning_rate=learning_rate)

    # 模态选择
    shapley_values = [communication.compute_shapley_values([local_model.predict(data[client_index], i) for i in range(num_modalities)], labels[client_index]) for client_index in range(num_clients)]
    selected_modalities = [communication.select_modalities(shapley_values[client_index], np.random.rand(), np.random.rand(), gamma) for client_index in range(num_clients)]

    # 客户端选择与服务器聚合
    selected_clients = communication.select_clients(local_losses, delta)
    global_weights = np.random.rand(num_modalities)

    #    确保每个客户端和模态的数据样本数量一致
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

    # 本地部署
    global_models = global_model.get_global_models()
    for client_index, local_model in enumerate(local_models):
        for modality_index in range(num_modalities):
            local_model.update_ensemble_model(global_models, labels[client_index])

# 更新模态模型和集成模型
final_global_models = global_model.get_global_models()
final_local_models = [local_model.save_model() for local_model in local_models]

print("算法运行完成，模型已更新。")