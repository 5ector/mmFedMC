import numpy as np

class Communication:
    def __init__(self, num_modalities):
        self.num_modalities = num_modalities

    def compute_shapley_values(self, predictions, true_labels):
        # 计算每个模态的Shapley值
        shapley_values = np.random.rand(len(predictions))  # Placeholder，实际需要计算Shapley值
        return shapley_values

    def select_modalities(self, shapley_values, model_size, recency, top_k):
        # 将Shapley值、模型大小和更新频率标准化
        normalized_shapley_values = (shapley_values - np.min(shapley_values)) / (np.max(shapley_values) - np.min(shapley_values))
        normalized_model_size = (model_size - np.min(model_size)) / (np.max(model_size) - np.min(model_size))
        normalized_recency = recency / np.max(recency)
        
        # 计算优先级
        priority = 0.5 * normalized_shapley_values + 0.3 * (1 - normalized_model_size) + 0.2 * normalized_recency
        selected_modalities = np.argsort(priority)[-top_k:]
        return selected_modalities

    def select_clients(self, local_losses, delta, K):
        # 根据损失值选择客户端
        num_clients = len(local_losses)
        threshold = int(np.ceil(delta * num_clients))
        top_losses_indices = np.argsort(local_losses)[-threshold:]
        selected_clients = top_losses_indices[:K]
        return selected_clients

    def aggregate_predictions(self, predictions):
        # 聚合所有模态模型的预测结果
        from scipy.stats import mode
        return mode(predictions, axis=0)[0]

    def update_models_stage_1(self, local_models, data_sizes):
        # 调用全局模型的 Shapley 值计算和聚合方法
        shapley_values = self.global_model.compute_and_aggregate_shapley(local_models, data_sizes)
        return shapley_values

    def update_models_stage_2(self, global_models, local_data, local_labels):
        # 更新本地集成模型
        self.local_model.update_ensemble_model_stage_2(global_models, local_data, local_labels)


    def send_model(self, model):
        # 将模型发送到服务器
        pass

# Example usage:
# communication = Communication(num_modalities=3)
# shapley_values = communication.compute_shapley_values(predictions, true_labels)
# selected_modalities = communication.select_modalities(shapley_values, model_size, recency, top_k=2)
# selected_clients = communication.select_clients(local_losses, top_delta=5)