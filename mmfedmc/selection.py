# selection.py
import numpy as np
class ModalitySelection:
    def __init__(self):
        # 初始化模态选择模块
        pass

    def select_modalities(self, priorities, gamma):
        """
        选择优先级最高的 gamma 个模态
        :param priorities: 每个模态的优先级评分
        :param gamma: 要选择的模态数量
        :return: 选定的模态索引
        """
        # 获取优先级最高的 gamma 个模态的索引
        top_gamma_indices = np.argsort(priorities)[-gamma:]
        return top_gamma_indices

class ClientSelection:
    def __init__(self):
        # 初始化客户端选择模块
        pass

    def select_clients(self, local_losses, delta, K):
        """
        选择要上传的客户端
        :param local_losses: 每个客户端的局部损失
        :param delta: 参数 delta
        :param K: 客户端总数
        :return: 选定的客户端索引
        """
        selected_clients = []
        num_clients = len(local_losses)

        for m in range(len(local_losses[0])):  # assuming local_losses is a list of lists with shape (num_clients, num_modalities)
            # 获取模态 m 的损失值
            modality_losses = [loss[m] for loss in local_losses]
            # 计算要选择的客户端数量
            threshold = int(np.ceil(delta * num_clients))
            # 获取损失值最高的客户端索引
            top_losses_indices = np.argsort(modality_losses)[-threshold:]
            selected_clients.extend(top_losses_indices)
        
        # 去重并返回
        return list(set(selected_clients))