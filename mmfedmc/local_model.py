import torch
from sklearn.linear_model import SGDClassifier
from mmfedmc.fusion import DecisionFusion

class LocalEnsembleModel:
    def __init__(self, num_modalities):
        self.models = [SGDClassifier() for _ in range(num_modalities)]
        self.ensemble_model = SGDClassifier()
        self.num_modalities = num_modalities
        self.decision_fusion = DecisionFusion()

    def train(self, data, labels, modality_index, epochs=1, learning_rate=0.01):
        for epoch in range(epochs):
            self.models[modality_index].partial_fit(data, labels, classes=torch.unique(labels).numpy())

    def predict(self, data, modality_index):
        return self.models[modality_index].predict(data)

    def update_ensemble_model(self, predictions, labels, epochs=1, learning_rate=0.01):
        for epoch in range(epochs):
            self.ensemble_model.partial_fit(predictions, labels, classes=torch.unique(labels).numpy())

    def update_ensemble_model_stage_1(self, predictions, shapley_values):
        fused_predictions = self.decision_fusion.stage_1_update(predictions, shapley_values)
        self.update_ensemble_model(fused_predictions, shapley_values)

    def update_ensemble_model_stage_2(self, global_models, local_data, local_labels):
        predictions = [global_models[m].predict(local_data) for m in range(self.num_modalities)]
        ensemble_predictions = self.fuse_predictions(predictions)
        self.update_ensemble_model(ensemble_predictions, local_labels)

    def fuse_predictions(self, predictions):
        fused_predictions = torch.mean(torch.stack(predictions), axis=0)
        return fused_predictions

    def calculate_mono_confidence(self, data, modality_index, true_labels):
        probabilities = self.models[modality_index].predict_proba(data)
        true_class_probabilities = probabilities[torch.arange(len(data)), true_labels]
        mono_confidence = torch.mean(true_class_probabilities)
        return mono_confidence

    def calculate_holo_confidence(self, data, true_labels):
        probabilities = [self.models[m].predict_proba(data) for m in range(self.num_modalities)]
        losses = [1 - torch.mean(prob[torch.arange(len(data)), true_labels]) for prob in probabilities]
        holo_confidence = [loss / sum(losses) for loss in losses]
        return holo_confidence

    def calculate_co_belief(self, data, true_labels):
        mono_confidences = [self.calculate_mono_confidence(data, m, true_labels) for m in range(self.num_modalities)]
        holo_confidences = self.calculate_holo_confidence(data, true_labels)
        co_belief = [mono_conf + holo_conf for mono_conf, holo_conf in zip(mono_confidences, holo_confidences)]
        return co_belief
    
    def calculate_distribution_uniformity(self, data, modality_index):
        probabilities = self.models[modality_index].predict_proba(data)
        C = probabilities.shape[1]
        mu = 1 / C
        du = torch.mean(torch.abs(probabilities - mu), axis=1)
        return torch.mean(du)

    def calculate_relative_calibration(self, data):
        DUs = [self.calculate_distribution_uniformity(data, m) for m in range(self.num_modalities)]
        RCs = [DU / sum(DUs) for DU in DUs]
        return RCs

    def calculate_calibrated_co_belief(self, data, true_labels):
        DUs = [self.calculate_distribution_uniformity(data, m) for m in range(self.num_modalities)]
        co_beliefs = self.calculate_co_belief(data, true_labels)
        RCs = self.calculate_relative_calibration(data)
        calibrated_co_beliefs = [co_belief * (rc if du < max(DUs) else 1) for co_belief, rc, du in zip(co_beliefs, RCs, DUs)]
        return calibrated_co_beliefs
        
    def save_model(self):
        pass

    def get_model_sizes(self):
        model_sizes = [self.models[m].coef_.size for m in range(self.num_modalities)]
        return model_sizes

# 示例用法:
# local_model = LocalEnsembleModel(num_modalities=3)
# local_model.train(data_modality_1, labels_modality_1, modality_index=0, epochs=5)
# local_model.train(data_modality_2, labels_modality_2, modality_index=1, epochs=5)
# local_model.train(data_modality_3, labels_modality_3, modality_index=2, epochs=5)
# predictions = [local_model.predict(data, i) for i in range(3)]
# local_model.update_ensemble_model(predictions, labels, epochs=5)