# communication.py

class Communication:
    def __init__(self, num_modalities):
        self.num_modalities = num_modalities

    def compute_shapley_values(self, predictions, true_labels):
        # 计算每个模态的Shapley值
        shapley_values = np.random.rand(len(predictions))  # Placeholder
        return shapley_values

    def select_modalities(self, shapley_values, model_size, recency, top_k):
        # 选择上传优先级最高的模态模型
        priority = shapley_values / (model_size + recency)
        selected_modalities = np.argsort(priority)[-top_k:]
        return selected_modalities

    def select_clients(self, local_losses, top_delta):
        # 选择损失较低的客户端进行上传
        selected_clients = np.argsort(local_losses)[:top_delta]
        return selected_clients

    def aggregate_predictions(self, predictions):
        # 聚合所有模态模型的预测结果
        from scipy.stats import mode
        return mode(predictions, axis=0)[0]

    def send_model(self, model):
        # 将模型发送到服务器
        pass

# Example usage:
# communication = Communication(num_modalities=3)
# shapley_values = communication.compute_shapley_values(predictions, true_labels)
# selected_modalities = communication.select_modalities(shapley_values, model_size, recency, top_k=2)
# selected_clients = communication.select_clients(local_losses, top_delta=5)