import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class BinaryMLP(nn.Module):
    def __init__(self):
        super(BinaryMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(97, 64),            
            nn.ReLU(),
            nn.Dropout(0.3),              
            nn.BatchNorm1d(64),            
            
            nn.Linear(64, 32),             
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 16),             
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 1)               
        )
        
    def forward(self, x):
        return self.model(x)
    

def train(model, loader, criterion, optimizer, num_epochs, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for epoch in range(num_epochs):
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)  # Ensure correct shape

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

            # Calculate predictions and update accuracy metrics
            preds = torch.sigmoid(outputs).round()  # Threshold at 0.5 for binary classification
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate overall metrics
    avg_loss = total_loss / (num_epochs * len(loader))  # Average loss over all batches
    overall_accuracy = correct_predictions / total_samples
    return overall_accuracy

def evaluate(model, testloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()  # Assuming binary classification with logits output

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are floats for BCELoss
            outputs = model(inputs)

            # Calculate loss for the batch
            batch_loss = criterion(outputs, labels.unsqueeze(1))  # Labels reshaped for compatibility
            total_loss += batch_loss.item()

            # Convert logits to predictions
            preds = torch.sigmoid(outputs).round()  # Threshold at 0.5 for binary classification
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Average loss over all batches
    avg_loss = total_loss / len(testloader)

    return avg_loss, accuracy, precision, recall, f1