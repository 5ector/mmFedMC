import torch

class GlobalModel:
    def __init__(self, num_modalities):
        self.global_models = [None] * num_modalities
        
    def compute_and_aggregate_shapley(self, local_models, data_sizes):
        # 计算 Shapley 值并聚合全局模态模型
        shapley_values = self.compute_shapley_values(local_models, data_sizes)
        self.aggregate_models(local_models, data_sizes)
        return shapley_values
    
    def aggregate_models(self, local_models, data_sizes):
        """
        根据样本数加权聚合模型参数
        :param local_models: 每个客户端的本地模型参数
        :param data_sizes: 每个客户端的样本数量
        """
        aggregated_models = []
        for m in range(len(local_models[0])):  # 遍历每个模态
            weighted_sum = torch.sum(torch.stack([data_sizes[k] * local_models[k][m] for k in range(len(local_models))]), dim=0)
            aggregated_models.append(weighted_sum / torch.sum(data_sizes))
        self.global_models = aggregated_models
    
    def get_global_models(self):
        return self.global_models

# 示例用法:
# global_model = GlobalModel(num_modalities=3)
# global_model.aggregate_models(local_models, weights)
# global_models = global_model.get_global_models()