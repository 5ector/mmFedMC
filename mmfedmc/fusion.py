# fusion.py

class DecisionFusion:
    def __init__(self, modalities):
        # 初始化决策融合模块
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.fusion_weights = np.random.rand(self.num_modalities)
        self.fusion_weights /= np.sum(self.fusion_weights)  # Normalize weights

    def fuse(self, predictions):
        # 进行决策级融合
        fused_output = 0
        for i, prediction in enumerate(predictions):
            fused_output += self.fusion_weights[i] * prediction
        return fused_output
    
    def stage_1_update(self, predictions, shapley_values):
        # 第一阶段更新，仅用于计算Shapley值
        return self.fuse(predictions)

    def stage_2_update(self, global_models, local_data):
        # 第二阶段更新，使用全局模态模型和本地数据更新集成模型
        global_predictions = [global_model.predict(local_data) for global_model in global_models]
        return self.fuse(global_predictions)

    def compute_fusion_weights(self, train_data, train_labels):
        # 动态计算融合权重
        self.fusion_weights = np.random.rand(self.num_modalities)
        self.fusion_weights /= np.sum(self.fusion_weights)  # Normalize weights
    
    def multimodal_fusion(self, data_point, models):
        # 决策级多模态融合
        fused_output = 0
        for i, modality in enumerate(self.modalities):
            modality_data = data_point[modality]
            modality_output = models[modality].predict(modality_data)  # Assuming models are pre-trained
            fused_output += self.fusion_weights[i] * modality_output
        return fused_output

