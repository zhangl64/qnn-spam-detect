import pennylane as qml
from pennylane import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef

import wandb
import numpy as np
import pandas as pd
import random
import os 
from dataset import download_trec2007_dataset, download_spamfilter_dataset
# Download datasets

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set all seeds here
seed = 42
set_seed(seed)
# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize WandB
wandb.init(project="Trec_Quantum-Spam-classification", config={
    "epochs": 10, # 
    "batch_size": 16,
    "learning_rate": 0.0005,
    "n_qubits": 8, # from 2 to 4 
    "k_folds": 5,
    "n_quantum_layers": 3,
})
config = wandb.config


#df = pd.read_csv("Put the path to the dataset here /processed_data.csv")
# df= pd.read_csv("Path to Kaggle dataset/spam-filter/versions/1/emails.csv")

# Please update the function to download the dataset from Kaggle
df = download_trec2007_dataset()
# df = download_spamfilter_dataset()

df = df.dropna(subset=['message'])
# df = df.dropna(subset=['text'])

# print(data.shape)
# print(data.head())
labels = df["label"]
texts = df["message"]
# texts = df["text"]
print(df["label"].value_counts())

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Convert to PyTorch Dataset
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)  # Move tensors to the device
        self.y = torch.tensor(y, dtype=torch.long).to(device)     # Move tensors to the device
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Quantum Circuit
n_qubits = config.n_qubits
dev = qml.device("default.qubit", wires=n_qubits)



class HybridModel(torch.nn.Module):
    def __init__(self, n_qubits, n_quantum_layers=3, n_classical_layers=2, output_dim=2):
        super(HybridModel, self).__init__()
        # Quantum circuit layers
        self.quantum_circuit = qml.qnn.TorchLayer(self.create_quantum_circuit(n_qubits, n_quantum_layers), {"weights": (n_quantum_layers, n_qubits)})
        # Additional classical neural network layers
        self.classical_layers = torch.nn.ModuleList([torch.nn.Linear(n_qubits if i == 0 else 10, 10) for i in range(n_classical_layers)])
        self.output_layer = torch.nn.Linear(10, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Quantum circuit output
        x = self.quantum_circuit(x)
        # Pass through classical layers
        for layer in self.classical_layers:
            x = self.relu(layer(x))
        # Output layer
        x = self.output_layer(x)
        return x
    
    def create_quantum_circuit(self, n_qubits, n_layers):
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # Initial angle embedding
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            # Additional layers
            for _ in range(n_layers):
                qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
                # You can add more types of quantum gates here for complexity
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return circuit

def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    mcc = matthews_corrcoef(all_targets, all_preds)  # Compute MCC

    return running_loss / len(train_loader), 100 * correct / total, precision, recall, f1, mcc

def validate_one_epoch(model, criterion, test_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')
    mcc = matthews_corrcoef(all_targets, all_preds)  # Compute MCC

    return running_loss / len(test_loader), 100 * correct / total, precision, recall, f1, mcc


# K-Fold Cross Validation
kfold = KFold(n_splits=config.k_folds, shuffle=True, random_state=42)
epochs = config.epochs

# Initialize lists to store loss/accuracy across folds
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_mccs, val_mccs = [], []  # Additional lists for MCC scores

for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
    print(f"Fold {fold + 1}")
    wandb.log({"fold": fold + 1})

    # Split texts and labels into training and validation sets for this fold
    X_train_fold, X_val_fold = texts.iloc[train_idx], texts.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Apply TfidfVectorizer to training data and transform validation data using the same vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_fold = vectorizer.fit_transform(X_train_fold).toarray()
    X_val_fold = vectorizer.transform(X_val_fold).toarray()

    # Apply PCA to reduce dimensions for training data and transform validation data using the same PCA
    pca = PCA(n_components=n_qubits)
    X_train_fold = pca.fit_transform(X_train_fold)
    X_val_fold = pca.transform(X_val_fold)

    # Create datasets and loaders for this fold
    train_dataset = SpamDataset(X_train_fold, y_train_fold)
    val_dataset = SpamDataset(X_val_fold, y_val_fold)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model and optimizer for each fold
    model = HybridModel(n_qubits=n_qubits, n_quantum_layers=3, n_classical_layers=2, output_dim=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Track loss and accuracy for each epoch
    fold_train_losses, fold_val_losses = [], []
    fold_train_accuracies, fold_val_accuracies = [], []
    fold_train_precisions, fold_val_precisions = [], []
    fold_train_recalls, fold_val_recalls = [], []
    fold_train_f1s, fold_val_f1s = [], []
    fold_train_mccs, fold_val_mccs = [], []  # Tracking MCC for each fold

    for epoch in range(epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_mcc = train_one_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_mcc = validate_one_epoch(model, criterion, val_loader)

        # Log metrics to WandB for each epoch and fold
        wandb.log({
            f"fold_{fold+1}_train_loss": train_loss,
            f"fold_{fold+1}_val_loss": val_loss,
            f"fold_{fold+1}_train_acc": train_acc,
            f"fold_{fold+1}_val_acc": val_acc,
            f"fold_{fold+1}_train_precision": train_prec,
            f"fold_{fold+1}_val_precision": val_prec,
            f"fold_{fold+1}_train_recall": train_rec,
            f"fold_{fold+1}_val_recall": val_rec,
            f"fold_{fold+1}_train_f1": train_f1,
            f"fold_{fold+1}_val_f1": val_f1,
            f"fold_{fold+1}_train_mcc": train_mcc,  # Logging MCC
            f"fold_{fold+1}_val_mcc": val_mcc,  # Logging MCC
            "epoch": epoch + 1
        })

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)
        fold_train_accuracies.append(train_acc)
        fold_val_accuracies.append(val_acc)
        fold_train_precisions.append(train_prec)
        fold_val_precisions.append(val_prec)
        fold_train_recalls.append(train_rec)
        fold_val_recalls.append(val_rec)
        fold_train_f1s.append(train_f1)
        fold_val_f1s.append(val_f1)
        fold_train_mccs.append(train_mcc)  # Store MCC
        fold_val_mccs.append(val_mcc)  # Store MCC

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Store metrics across folds
    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)
    train_accuracies.append(fold_train_accuracies)
    val_accuracies.append(fold_val_accuracies)
    train_precisions.append(fold_train_precisions)
    val_precisions.append(fold_val_precisions)
    train_recalls.append(fold_train_recalls)
    val_recalls.append(fold_val_recalls)
    train_f1s.append(fold_train_f1s)
    val_f1s.append(fold_val_f1s)
    train_mccs.append(fold_train_mccs)  # Append fold MCCs
    val_mccs.append(fold_val_mccs)  # Append fold MCCs

    # Save the model for each fold (optional)
    wandb.watch(model, log="all")

# Calculate average and standard deviation of the metrics across folds
avg_train_loss = np.mean(train_losses, axis=0)
avg_val_loss = np.mean(val_losses, axis=0)
avg_train_acc = np.mean(train_accuracies, axis=0)
avg_val_acc = np.mean(val_accuracies, axis=0)
avg_train_prec = np.mean(train_precisions, axis=0)
avg_val_prec = np.mean(val_precisions, axis=0)
avg_train_rec = np.mean(train_recalls, axis=0)
avg_val_rec = np.mean(val_recalls, axis=0)
avg_train_f1 = np.mean(train_f1s, axis=0)
avg_val_f1 = np.mean(val_f1s, axis=0)
avg_train_mcc = np.mean(train_mccs, axis=0)  # MCC average
avg_val_mcc = np.mean(val_mccs, axis=0)  # MCC average

std_train_loss = np.std(train_losses, axis=0)
std_val_loss = np.std(val_losses, axis=0)
std_train_acc = np.std(train_accuracies, axis=0)
std_val_acc = np.std(val_accuracies, axis=0)
std_train_prec = np.std(train_precisions, axis=0)
std_val_prec = np.std(val_precisions, axis=0)
std_train_rec = np.std(train_recalls, axis=0)
std_val_rec = np.std(val_recalls, axis=0)
std_train_f1 = np.std(train_f1s, axis=0)
std_val_f1 = np.std(val_f1s, axis=0)
std_train_mcc = np.std(train_mccs, axis=0)  # MCC standard deviation
std_val_mcc = np.std(val_mccs, axis=0)  # MCC standard deviation

# Log the average and standard deviation across folds
for epoch in range(epochs):
    wandb.log({
        "avg_train_loss": avg_train_loss[epoch],
        "avg_val_loss": avg_val_loss[epoch],
        "avg_train_acc": avg_train_acc[epoch],
        "avg_val_acc": avg_val_acc[epoch],
        "avg_train_precision": avg_train_prec[epoch],
        "avg_val_precision": avg_val_prec[epoch],
        "avg_train_recall": avg_train_rec[epoch],
        "avg_val_recall": avg_val_rec[epoch],
        "avg_train_f1": avg_train_f1[epoch],
        "avg_val_f1": avg_val_f1[epoch],
        "avg_train_mcc": avg_train_mcc[epoch],  # Log MCC averages
        "avg_val_mcc": avg_val_mcc[epoch],  # Log MCC averages
        "std_train_loss": std_train_loss[epoch],
        "std_val_loss": std_val_loss[epoch],
        "std_train_acc": std_train_acc[epoch],
        "std_val_acc": std_val_acc[epoch],
        "std_train_precision": std_train_prec[epoch],
        "std_val_precision": std_val_prec[epoch],
        "std_train_recall": std_train_rec[epoch],
        "std_val_recall": std_val_rec[epoch],
        "std_train_f1": std_train_f1[epoch],
        "std_val_f1": std_val_f1[epoch],
        "std_train_mcc": std_train_mcc[epoch],  # Log MCC standard deviations
        "std_val_mcc": std_val_mcc[epoch],  # Log MCC standard deviations
        "epoch": epoch + 1
    })

# Finish WandB logging
wandb.finish()
