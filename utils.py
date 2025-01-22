import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from typing import List
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def weighted_average(metrics):
    accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
    precisions = [num_examples * m.get("precision", 0) for num_examples, m in metrics]
    recalls = [num_examples * m.get("recall", 0) for num_examples, m in metrics]
    f1_scores = [num_examples * m.get("f1_score", 0) for num_examples, m in metrics]
    
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0,
        "precision": sum(precisions) / sum(examples) if sum(examples) > 0 else 0,
        "recall": sum(recalls) / sum(examples) if sum(examples) > 0 else 0,
        "f1_score": sum(f1_scores) / sum(examples) if sum(examples) > 0 else 0,
    }
