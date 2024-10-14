# global_model.py
import numpy as np

class GlobalModel:
    def __init__(self, num_modalities):
        self.global_models = [None] * num_modalities

    def aggregate_models(self, local_models, weights):
        # 聚合本地模型，更新全局模型
        for m in range(len(self.global_models)):
            weighted_sum = np.sum([weights[k] * local_models[k][m] for k in range(len(local_models))], axis=0)
            self.global_models[m] = weighted_sum / np.sum(weights)

    def get_global_models(self):
        return self.global_models

# Example usage:
# global_model = GlobalModel(num_modalities=3)
# global_model.aggregate_models(local_models, weights)
# global_models = global_model.get_global_models()