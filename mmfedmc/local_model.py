# local_model.py
import numpy as np
from sklearn.linear_model import SGDClassifier
from mmfedmc.fusion import DecisionFusion

class LocalEnsembleModel:
    def __init__(self, num_modalities):
        self.models = [SGDClassifier() for _ in range(num_modalities)]
        self.ensemble_model = SGDClassifier()
        self.num_modalities = num_modalities

    def train(self, data, labels, modality_index, epochs=1, learning_rate=0.01):
        # 使用SGD优化器训练对应模态的模型
        for epoch in range(epochs):
            self.models[modality_index].partial_fit(data, labels, classes=np.unique(labels))

    def predict(self, data, modality_index):
        # 使用对应模态的模型进行预测
        return self.models[modality_index].predict(data)

    def update_ensemble_model(self, predictions, labels, epochs=1, learning_rate=0.01):
        # 使用SGD优化器更新集成模型
        for epoch in range(epochs):
            self.ensemble_model.partial_fit(predictions, labels, classes=np.unique(labels))
    
    def update_ensemble_model_stage_1(self, predictions, shapley_values):
        # 第一阶段更新，仅用于计算Shapley值
        fused_predictions = self.decision_fusion.stage_1_update(predictions, shapley_values)
        self.update_ensemble_model(fused_predictions, shapley_values)
    
    def update_ensemble_model_stage_2(self, global_models, local_data, local_labels):
        # 使用全局模态模型和本地数据更新个性化集成模型
        predictions = [global_models[m].predict(local_data) for m in range(self.num_modalities)]
        ensemble_predictions = self.fuse_predictions(predictions)
        self.update_ensemble_model(ensemble_predictions, local_labels)

    def fuse_predictions(self, predictions):
        # 决策级融合逻辑，可以根据需要调整
        fused_predictions = np.mean(predictions, axis=0)  # Placeholder
        return fused_predictions

    def save_model(self):
        # 保存本地集成模型
        pass
    
    def get_model_sizes(self):
        # 获取每个模态模型的参数大小
        model_sizes = [self.models[m].coef_.size for m in range(self.num_modalities)]
        return model_sizes

# Example usage:
# local_model = LocalEnsembleModel(num_modalities=3)
# local_model.train(data_modality_1, labels_modality_1, modality_index=0, epochs=5)
# local_model.train(data_modality_2, labels_modality_2, modality_index=1, epochs=5)
# local_model.train(data_modality_3, labels_modality_3, modality_index=2, epochs=5)
# predictions = [local_model.predict(data, i) for i in range(3)]
# local_model.update_ensemble_model(predictions, labels, epochs=5)