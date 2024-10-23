import torch

class ModalitySelection:
    def __init__(self):
        pass

    def select_modalities(self, priorities, gamma):
        """
        选择优先级最高的 gamma 个模态
        :param priorities: 每个模态的优先级评分
        :param gamma: 要选择的模态数量
        :return: 选定的模态索引
        """
        priorities_tensor = torch.tensor(priorities)
        top_gamma_indices = torch.topk(priorities_tensor, gamma).indices
        return top_gamma_indices.tolist()

class ClientSelection:
    def __init__(self):
        pass

    def select_clients(self, local_losses, delta, K):
        """
        选择要上传的客户端
        :param local_losses: 每个客户端的局部损失
        :param delta: 参数 delta
        :param K: 客户端总数
        :return: 选定的客户端索引
        """
        num_clients = len(local_losses)
        local_losses_tensor = torch.tensor(local_losses)
        
        selected_clients = set()
        
        for m in range(local_losses_tensor.size(1)):  # assuming local_losses is a 2D list with shape (num_clients, num_modalities)
            modality_losses = local_losses_tensor[:, m]
            threshold = int(torch.ceil(torch.tensor(delta * num_clients)))
            top_losses_indices = torch.topk(modality_losses, threshold).indices
            selected_clients.update(top_losses_indices.tolist())
        
        return list(selected_clients)

def select_modalities_and_clients(communication, local_model, data, labels, gamma, local_losses, delta, num_modalities, num_clients):
    shapley_values = [communication.compute_shapley_values([local_model.predict(data[client_index], i) for i in range(num_modalities)], labels[client_index]) for client_index in range(num_clients)]
    selected_modalities = [communication.select_modalities(shapley_values[client_index], torch.rand(num_modalities), torch.rand(num_modalities), gamma) for client_index in range(num_clients)]
    selected_clients = communication.select_clients(local_losses, delta)
    return selected_modalities, selected_clients