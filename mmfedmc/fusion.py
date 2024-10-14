# fusion.py

class DecisionFusion:
    def __init__(self):
        # 初始化决策融合模块
        pass

    def fuse(self, predictions):
        # 进行决策级融合
        return sum(predictions) / len(predictions)
    
    def stage_1_update(self, predictions, shapley_values):
        # 第一阶段更新，仅用于计算Shapley值
        return self.fuse(predictions)

    def stage_2_update(self, global_models, local_data):
        # 第二阶段更新，使用全局模态模型和本地数据更新集成模型
        global_predictions = [global_model.predict(local_data) for global_model in global_models]
        return self.fuse(global_predictions)