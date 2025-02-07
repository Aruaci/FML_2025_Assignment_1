from utils import set_parameters, get_parameters
from model import train, evaluate
from flwr.client import NumPyClient
import torch
import torch.nn as nn

class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, num_epochs, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_epochs = num_epochs
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        accuracy = train(self.model, self.trainloader, self.criterion, self.optimizer, self.num_epochs, self.device)
        return get_parameters(self.model), len(self.trainloader), {"accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy, precision, recall, f1 = evaluate(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }