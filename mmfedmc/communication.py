import numpy as np
from itertools import combinations

class Communication:
    def __init__(self, num_modalities, alpha_s=0.5, alpha_c=0.5):
        self.num_modalities = num_modalities
        self.alpha_s = alpha_s
        self.alpha_c = alpha_c
        self.last_upload_rounds = np.zeros((num_modalities,), dtype=int)  # 初始化每个模态的上次上传轮次

    def compute_shapley_values(self, predictions, true_labels):
        # 计算每个模态的Shapley值
        num_modalities = len(predictions)
        shapley_values = np.zeros(num_modalities)
        
        for m in range(num_modalities):
            for subset in combinations(range(num_modalities), m):
                subset_with_m = subset + (m,)
                prediction_without_m = self.ensemble_predict(predictions, subset)
                prediction_with_m = self.ensemble_predict(predictions, subset_with_m)
                
                marginal_contribution = np.mean(prediction_with_m == true_labels) - np.mean(prediction_without_m == true_labels)
                shapley_values[m] += marginal_contribution / (np.math.comb(num_modalities - 1, len(subset)) * num_modalities)
                
        return shapley_values
    
    def ensemble_predict(self, predictions, modalities):
        # 使用指定模态的预测进行集成预测
        combined_predictions = np.mean([predictions[m] for m in modalities], axis=0)
        return np.round(combined_predictions)

    def select_modalities(self, shapley_values, model_size, recency, top_k):
        # 将Shapley值、模型大小和更新频率标准化
        normalized_shapley_values = (shapley_values - np.min(shapley_values)) / (np.max(shapley_values) - np.min(shapley_values))
        normalized_model_size = (model_size - np.min(model_size)) / (np.max(model_size) - np.min(model_size))
        normalized_recency = recency / np.max(recency)
        
         # 计算优先级
        priority = self.alpha_s * normalized_shapley_values + self.alpha_c * (1 - normalized_model_size) + (1 - self.alpha_s - self.alpha_c) * normalized_recency
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
    
    def compute_recency(self, current_round, modality_index):
        # 计算指定模态的 recency 值
        recency = current_round - self.last_upload_rounds[modality_index] - 1
        return recency

    def update_last_upload_round(self, modality_index, current_round):
        # 更新指定模态的上次上传轮次
        self.last_upload_rounds[modality_index] = current_round

# 示例用法:
# communication = Communication(num_modalities=3, alpha_s=0.5, alpha_c=0.3)
# shapley_values = communication.compute_shapley_values(predictions, true_labels)
# model_size = [100, 200, 150]  # 示例模型大小
# recency = [communication.compute_recency(current_round=10, modality_index=i) for i in range(3)]
# selected_modalities = communication.select_modalities(shapley_values, model_size, recency, top_k=2)